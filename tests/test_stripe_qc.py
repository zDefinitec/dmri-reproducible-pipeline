from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.signal import convolve2d

from dmri_pipeline.stripe_qc import (
    QCDecision,
    STRIPE_KERNEL,
    StripeMetrics,
    StripeQCError,
    classify_csi,
    compute_stripe_indices,
    decide_qc,
    run_stripe_qc,
)


def _metrics(values: list[float]) -> StripeMetrics:
    count = len(values)
    return StripeMetrics(
        a_si=np.ones(count),
        c_si=np.asarray(values, dtype=float),
        shells=np.zeros(count, dtype=int),
        peak_sagittal=np.zeros(count, dtype=int),
    )


def _positive_volume(shape: tuple[int, int, int] = (4, 5, 6)) -> np.ndarray:
    y, z = np.indices(shape[1:])
    plane = 10.0 + y + z**2
    return np.stack([plane + x for x in range(shape[0])], axis=0)


def _write_qc_inputs(tmp_path: Path):
    from dmri_pipeline.config import load_config

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    base = _positive_volume((5, 6, 7))
    striped = base.copy()
    striped[:, :, 1::2] += 75.0
    data = np.stack([base, striped, base * 1.1], axis=-1).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), inputs / "raw_pa.nii.gz")
    np.savetxt(inputs / "raw_pa.bval", [[0.0, 40.0, 995.0]], fmt="%.1f")

    config_path = tmp_path / "anonymous.yaml"
    config_path.write_text(
        """subject_id: ANON001
inputs:
  dwi_pa: inputs/raw_pa.nii.gz
  bvals: inputs/raw_pa.bval
  bvecs: inputs/unused.bvec
  b0_ap: inputs/unused_ap.nii.gz
output_root: outputs
acquisition:
  pa_vector: [0, -1, 0]
  ap_vector: [0, 1, 0]
  total_readout_time: 0.07
analysis:
  ambiguous_qc_reviewed: false
""",
        encoding="utf-8",
    )
    return load_config(config_path)


def test_kernel_matches_henrique_matlab_definition():
    expected = np.tile([0.0, -1.0, 2.0, -1.0, 0.0], (5, 1))
    np.testing.assert_array_equal(STRIPE_KERNEL, expected)


@pytest.mark.parametrize(
    ("values", "reviewed", "status"),
    [
        ([1.0] * 10, False, "INCLUDE"),
        ([1.26, 1.0, 1.0], False, "INCLUDE_WITH_FLAGS"),
        ([1.26] * 5, False, "EXCLUDE"),
        ([1.15, 1.0], False, "HOLD_FOR_REVIEW"),
        ([1.15, 1.0], True, "INCLUDE_AFTER_REVIEW"),
    ],
)
def test_qc_gate(values, reviewed, status):
    assert decide_qc(_metrics(values), reviewed).status == status


@pytest.mark.parametrize(
    ("value", "classification"),
    [
        (1.0, "normal"),
        (1.149999, "normal"),
        (1.15, "ambiguous"),
        (1.25, "ambiguous"),
        (1.250001, "high"),
    ],
)
def test_classify_csi_obeys_binding_boundaries(value, classification):
    assert classify_csi(value) == classification


def test_decision_boundaries_use_zero_based_indices_and_exit_codes():
    decision = decide_qc(_metrics([1.15, 1.25, 1.250001]), False)
    assert decision.status == "HOLD_FOR_REVIEW"
    assert decision.high_indices == (2,)
    assert decision.ambiguous_indices == (0, 1)
    assert decision.exit_code == 21


def test_exclusion_takes_precedence_over_reviewed_ambiguity():
    decision = decide_qc(_metrics([1.26] * 5 + [1.2]), True)
    assert decision == QCDecision(
        status="EXCLUDE",
        high_indices=(0, 1, 2, 3, 4),
        ambiguous_indices=(5,),
        exit_code=20,
    )


def test_review_flag_without_ambiguity_does_not_change_clean_decision():
    assert decide_qc(_metrics([1.0, 1.0]), True).status == "INCLUDE"


def test_metrics_are_validated_immutable_and_defensively_copied():
    source = np.array([2.0, 4.0])
    metrics = StripeMetrics(
        a_si=source,
        c_si=np.array([1.0, 2.0]),
        shells=np.array([0, 0]),
        peak_sagittal=np.array([1, 2]),
    )
    source[0] = 99.0
    assert metrics.a_si[0] == 2.0
    with pytest.raises(ValueError):
        metrics.c_si[0] = 3.0
    with pytest.raises(ValueError):
        metrics.a_si.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        metrics.shells = np.array([1, 1])
    with pytest.raises(StripeQCError, match="same nonzero length"):
        StripeMetrics(
            a_si=np.ones(2),
            c_si=np.ones(1),
            shells=np.zeros(2),
            peak_sagittal=np.zeros(2),
        )


def test_decisions_are_validated_and_immutable():
    indices = [2, 4]
    decision = QCDecision("INCLUDE_WITH_FLAGS", indices, (), 0)
    indices[0] = 99
    assert decision.high_indices == (2, 4)
    with pytest.raises(FrozenInstanceError):
        decision.status = "INCLUDE"
    with pytest.raises(StripeQCError, match="exit code"):
        QCDecision("EXCLUDE", (), (), 0)


def test_shell_rounding_and_within_shell_normalization():
    base = _positive_volume()
    data = np.stack([base, base * 2.0, base * 3.0, base * 6.0], axis=-1)
    metrics = compute_stripe_indices(data, [49.0, 51.0, 1049.0, 1051.0])
    np.testing.assert_array_equal(metrics.shells, [0, 100, 1000, 1100])
    np.testing.assert_allclose(metrics.c_si, np.ones(4))

    paired = compute_stripe_indices(
        np.stack([base, base * 2.0, base * 3.0], axis=-1),
        [995.0, 1004.0, 2001.0],
    )
    np.testing.assert_array_equal(paired.shells, [1000, 1000, 2000])
    np.testing.assert_allclose(paired.c_si, [1.0, 2.0, 1.0])


def test_exact_calculation_matches_independent_full_convolution():
    volume = _positive_volume((3, 4, 5))
    metrics = compute_stripe_indices(volume[..., None], [1000.0])
    responses = [
        np.abs(convolve2d(volume[x, :, :], STRIPE_KERNEL, mode="full")).sum()
        for x in range(volume.shape[0])
    ]
    assert metrics.a_si[0] == pytest.approx(sum(responses))
    assert metrics.peak_sagittal[0] == int(np.argmax(responses))
    assert metrics.c_si[0] == pytest.approx(1.0)


def test_nibabel_proxy_input_matches_numpy_input(tmp_path):
    data = np.stack([_positive_volume(), _positive_volume() * 1.4], axis=-1)
    path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    image = nib.load(path)
    from_proxy = compute_stripe_indices(image, [1000.0, 1000.0])
    from_array = compute_stripe_indices(data, [1000.0, 1000.0])
    np.testing.assert_allclose(from_proxy.a_si, from_array.a_si)
    np.testing.assert_array_equal(from_proxy.peak_sagittal, from_array.peak_sagittal)


def test_deterministic_synthetic_stripe_exceeds_threshold():
    base = _positive_volume((5, 8, 9))
    striped = base.copy()
    striped[:, :, 1::2] += 100.0
    metrics = compute_stripe_indices(
        np.stack([base, striped], axis=-1), [1000.0, 1000.0]
    )
    assert metrics.c_si[1] > metrics.c_si[0]
    assert metrics.c_si[1] > 1.25


@pytest.mark.parametrize(
    ("image", "bvals", "message"),
    [
        (np.ones((2, 2, 2)), [0.0], "4D"),
        (np.ones((2, 2, 2, 2)), [0.0], "count"),
        (np.ones((2, 2, 2, 1)), [np.nan], "finite"),
        (np.full((2, 2, 2, 1), np.nan), [0.0], "finite"),
        (np.ones((2, 2, 2, 0)), [], "empty"),
    ],
)
def test_malformed_inputs_fail_actionably(image, bvals, message):
    with pytest.raises(StripeQCError, match=message):
        compute_stripe_indices(image, bvals)


def test_zero_shell_minimum_fails_actionably():
    with pytest.raises(StripeQCError, match="nonpositive"):
        compute_stripe_indices(np.zeros((3, 3, 3, 2)), [1000.0, 1000.0])


def test_classification_rejects_nonfinite_values():
    with pytest.raises(StripeQCError, match="finite"):
        classify_csi(np.inf)


def test_run_writes_required_generic_outputs_and_metadata(tmp_path):
    config = _write_qc_inputs(tmp_path)
    output_dir = tmp_path / "qc"

    decision = run_stripe_qc(config, output_dir)

    expected_files = {
        "stripe_metrics.csv",
        "stripe_decision.json",
        "automatic_summary.txt",
        "00_raw_b0_anatomy_overview.png",
        "01_cSI_by_volume.png",
        "02_cSI_by_shell.png",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    assert list(output_dir.glob("03_candidate_details_*.png"))
    assert list(output_dir.glob("04_all_volumes_*.png"))
    assert decision.status in {
        "INCLUDE",
        "INCLUDE_WITH_FLAGS",
        "HOLD_FOR_REVIEW",
        "INCLUDE_AFTER_REVIEW",
        "EXCLUDE",
    }

    payload = json.loads((output_dir / "stripe_decision.json").read_text())
    assert payload["method"] == "Henrique Appendix A sagittal stripe index"
    assert payload["thresholds"] == {
        "ambiguous_min_inclusive": 1.15,
        "high_min_exclusive": 1.25,
        "exclude_high_volume_count": 5,
    }
    assert payload["decision"] == decision.status
    assert payload["exit_code"] == decision.exit_code
    assert payload["volume_counts"]["total"] == 3
    assert payload["flagged_indices_zero_based"]["high"] == list(
        decision.high_indices
    )
    assert payload["flagged_volume_numbers_one_based"]["high"] == [
        index + 1 for index in decision.high_indices
    ]
    assert payload["maximum_csi"]["volume_index_zero_based"] in (0, 1, 2)
    assert payload["maximum_csi"]["volume_number_one_based"] in (1, 2, 3)
    assert payload["shell_counts"] == {"0": 2, "1000": 1}
    assert "not computed" in payload["cohort_fsi"].lower()

    with (output_dir / "stripe_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["volume_index_zero_based"]) for row in rows] == [0, 1, 2]
    assert [int(row["volume_number_one_based"]) for row in rows] == [1, 2, 3]
    assert {row["classification"] for row in rows} <= {
        "normal",
        "ambiguous",
        "high",
    }

    summary = (output_dir / "automatic_summary.txt").read_text(encoding="utf-8")
    assert "ANON001" in summary
    assert "one-based" in summary
    assert "cohort fsi was not computed" in summary.lower()
    assert all(path.stat().st_size > 0 for path in output_dir.iterdir())


def test_run_handles_one_b0_and_partial_all_volume_sheet(tmp_path):
    config = _write_qc_inputs(tmp_path)
    bvals = np.array([0.0, 1000.0, 1000.0])
    np.savetxt(config.bvals, bvals[None, :], fmt="%.0f")
    output_dir = tmp_path / "one_b0_qc"
    run_stripe_qc(config, output_dir)
    assert (output_dir / "00_raw_b0_anatomy_overview.png").is_file()
    assert len(list(output_dir.glob("04_all_volumes_*.png"))) == 1


def test_run_refuses_to_overwrite_existing_output_directory(tmp_path):
    config = _write_qc_inputs(tmp_path)
    output_dir = tmp_path / "already_exists"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(StripeQCError, match="already exists"):
        run_stripe_qc(config, output_dir)

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_run_accepts_stage_runner_created_empty_output_directory(tmp_path):
    config = _write_qc_inputs(tmp_path)
    output_dir = tmp_path / "stage_work"
    output_dir.mkdir()

    decision = run_stripe_qc(config, output_dir)

    assert decision.status in {
        "INCLUDE",
        "INCLUDE_WITH_FLAGS",
        "INCLUDE_AFTER_REVIEW",
        "HOLD_FOR_REVIEW",
        "EXCLUDE",
    }
    assert (output_dir / "stripe_decision.json").is_file()


def test_outputs_are_deterministic_for_identical_inputs(tmp_path):
    config = _write_qc_inputs(tmp_path)
    first = tmp_path / "qc_first"
    second = tmp_path / "qc_second"
    run_stripe_qc(config, first)
    run_stripe_qc(replace(config, output_root=tmp_path / "other_outputs"), second)
    for name in ("stripe_metrics.csv", "stripe_decision.json", "automatic_summary.txt"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
