from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import dmri_pipeline
import dmri_pipeline.orchestrator as orchestrator
import dmri_pipeline.stripe_qc as stripe_qc
from dmri_pipeline.orchestrator import (
    STAGE_ORDER,
    PipelineOutcome,
    _SubjectLock,
    _dry_run,
    _dry_run_commands,
    _installed_memory_gib,
    _software_provenance,
    _validate_eddy_outputs,
    _validate_noddi_outputs,
    _validate_subject_atlas,
    _validate_topup_outputs,
    _write_canonical_bvecs,
    _write_eddy_quad_json,
    build_plan,
    run_pipeline,
)
from dmri_pipeline.audit import audit_inputs
from dmri_pipeline.fsl import FSLInstallation
from dmri_pipeline.noddi import MATLABInstallation, NODDIError
from dmri_pipeline.state import StageContext, StageRunner, StageSpec, StageStateError
from dmri_pipeline.stripe_qc import StripeMetrics


EXPECTED_ORDER = (
    "00_input_audit",
    "00_pre_denoise_motion_qc",
    "01_denoise",
    "02_gibbs",
    "03_topup",
    "04_bet",
    "05_eddy",
    "06_dti",
    "07_dki",
    "07_dki_direct",
    "08_noddi",
    "09_jhu_48roi",
    "10_summary",
    "qc",
    "report",
)


@pytest.fixture(autouse=True)
def _isolate_subject_lock_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_lock_anchor_root",
        lambda: tmp_path / "lock-anchors",
    )


def test_stage_order_is_exact_and_public() -> None:
    assert STAGE_ORDER == EXPECTED_ORDER
    assert dmri_pipeline.STAGE_ORDER == EXPECTED_ORDER
    assert dmri_pipeline.build_plan is build_plan


def test_build_plan_returns_real_specs_in_exact_order(subject_config) -> None:
    plan = build_plan(subject_config)

    assert all(isinstance(stage, StageSpec) for stage in plan)
    assert tuple(stage.name for stage in plan) == EXPECTED_ORDER
    assert plan[0].input_paths == (
        subject_config.dwi_pa,
        subject_config.bvals,
        subject_config.bvecs,
        subject_config.b0_ap,
    )
    assert all(
        path.is_file()
        for stage in plan
        for path in (*stage.source_paths, *stage.resource_paths)
    )


def test_missing_explicit_plan_source_has_typed_dependency_semantics(
    subject_config, tmp_path: Path
) -> None:
    spec = StageSpec(
        "synthetic",
        lambda work: None,
        lambda work: (),
        (subject_config.bvals,),
        (tmp_path / "missing-source.py",),
    )

    with pytest.raises(orchestrator.PipelineDependencyError, match="missing"):
        orchestrator._validate_plan_sources((spec,))


def test_build_plan_wires_canonical_and_rotated_gradients(subject_config) -> None:
    plan = {stage.name: stage for stage in build_plan(subject_config)}
    root = subject_config.subject_output

    assert root / "00_input_audit" / "bvecs_fsl_3xN" in plan["05_eddy"].input_paths
    rotated = root / "05_eddy" / "eddy_unwarped_images.eddy_rotated_bvecs"
    for name in ("06_dti", "07_dki", "07_dki_direct", "08_noddi"):
        assert rotated in plan[name].input_paths


@pytest.mark.parametrize("transpose", [False, True])
def test_canonical_bvec_materialization_is_finite_3_by_n(
    subject_config, tmp_path: Path, transpose: bool
) -> None:
    values = np.loadtxt(subject_config.bvecs)
    source = tmp_path / "vectors.txt"
    np.savetxt(source, values.T if transpose else values)
    destination = tmp_path / "canonical.bvec"

    _write_canonical_bvecs(source, destination, values.shape[1])

    actual = np.loadtxt(destination)
    assert actual.shape == (3, values.shape[1])
    np.testing.assert_allclose(actual, values)
    assert np.isfinite(actual).all()


def test_canonical_bvec_rejects_nonfinite_values(tmp_path: Path) -> None:
    source = tmp_path / "vectors.txt"
    source.write_text("0 0\n1 nan\n0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        _write_canonical_bvecs(source, tmp_path / "out", 2)


def test_subject_atlas_requires_same_grid_and_exact_labels(tmp_path: Path) -> None:
    affine = np.eye(4)
    reference = tmp_path / "fa.nii.gz"
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 3), dtype=np.float32), affine), reference)
    values = np.arange(1, 49, dtype=np.int16).reshape(4, 4, 3)
    nib.save(nib.Nifti1Image(values, affine), atlas)

    _validate_subject_atlas(atlas, reference)

    values[0, 0, 0] = 0
    nib.save(nib.Nifti1Image(values, affine), atlas)
    with pytest.raises(ValueError, match="exactly 1 through 48"):
        _validate_subject_atlas(atlas, reference)


def test_subject_lock_is_nonblocking_and_rejects_second_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_lock_anchor_root",
        lambda: tmp_path / "lock-anchors",
        raising=False,
    )
    subject = tmp_path / "subject"
    subject.mkdir()
    with _SubjectLock(subject):
        with pytest.raises(StageStateError, match="already running"):
            with _SubjectLock(subject):
                pass


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MemTotal:       16777216 kB\nMemFree: 1 kB\n", 16777216),
        ("MemFree: 1 kB\nMemTotal: 8388608 kB\n", 8388608),
    ],
)
def test_parse_memtotal_kib(text: str, expected: int) -> None:
    assert orchestrator._parse_memtotal_kib(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MemFree: 1 kB\n",
        "MemTotal: 0 kB\n",
        "MemTotal: -1 kB\n",
        "MemTotal: unknown kB\n",
        "MemTotal: 1 MB\n",
        "MemTotal: 1 kB\nMemTotal: 2 kB\n",
        "MemTotal: unknown kB\nMemTotal: 1 kB\n",
        "MemTotal:\n1 kB\n",
    ],
)
def test_parse_memtotal_kib_rejects_malformed_values(text: str) -> None:
    with pytest.raises(NODDIError, match="MemTotal"):
        orchestrator._parse_memtotal_kib(text)


def test_installed_memory_reads_proc_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16777216 kB\n", encoding="utf-8")

    assert _installed_memory_gib(meminfo) == 16.0


def test_pipeline_outcome_is_immutable(subject_config) -> None:
    outcome = PipelineOutcome(
        subject=subject_config.subject_id,
        status="COMPLETE",
        stages=(),
        subject_output=subject_config.subject_output,
    )
    with pytest.raises(FrozenInstanceError):
        outcome.status = "FAILED"  # type: ignore[misc]


def test_exclusion_stops_before_external_dependency_discovery(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(orchestrator.__file__)
    audit_output = subject_config.subject_output / "00_input_audit" / "audit.txt"

    def first_action(work: Path) -> None:
        (work / "audit.txt").write_text("valid\n", encoding="utf-8")

    def stripe_action(work: Path) -> None:
        payload = {
            "subject_id": subject_config.subject_id,
            "decision": "EXCLUDE",
            "exit_code": 20,
            "ambiguous_reviewed": False,
            "flagged_indices_zero_based": {
                "high": [0, 1, 2, 3, 4],
                "ambiguous": [],
            },
        }
        (work / "stripe_decision.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )

    plan = [
        StageSpec(
            "00_input_audit",
            first_action,
            lambda work: (work / "audit.txt",),
            (subject_config.dwi_pa,),
            (source,),
        ),
        StageSpec(
            "00_pre_denoise_motion_qc",
            stripe_action,
            lambda work: (work / "stripe_decision.json",),
            (audit_output,),
            (source,),
        ),
    ]
    monkeypatch.setattr(orchestrator, "_build_plan", lambda config, runtime: plan)
    monkeypatch.setattr(
        orchestrator,
        "_discover_runtime",
        lambda config: pytest.fail("external dependency discovery ran before QC gate"),
    )

    outcome = run_pipeline(subject_config, "run")

    assert outcome.status == "EXCLUDED"
    assert outcome.stage_statuses == (
        ("00_input_audit", "completed"),
        ("00_pre_denoise_motion_qc", "completed"),
    )


def test_manual_review_transition_rejects_forged_outputs_without_records(
    subject_config,
) -> None:
    root = subject_config.subject_output
    audit_dir = root / "00_input_audit"
    stripe_dir = root / "00_pre_denoise_motion_qc"
    audit_dir.mkdir(parents=True)
    stripe_dir.mkdir()
    (audit_dir / "input_audit.json").write_text(
        __import__("json").dumps(audit_inputs(subject_config).to_dict()),
        encoding="utf-8",
    )
    (stripe_dir / "stripe_decision.json").write_text(
        __import__("json").dumps(
            {
                "subject_id": subject_config.subject_id,
                "decision": "HOLD_FOR_REVIEW",
                "ambiguous_reviewed": False,
            }
        ),
        encoding="utf-8",
    )
    reviewed = replace(
        subject_config,
        analysis=replace(
            subject_config.analysis,
            ambiguous_qc_reviewed=True,
        ),
    )
    runner = StageRunner(
        StageContext(reviewed, Path(__file__).parents[1], root, {"python": "test"})
    )

    assert not orchestrator._safe_manual_review_transition(
        reviewed, runner, audit_inputs(reviewed)
    )


def test_manual_review_false_to_true_accepts_genuine_qc_hold_records(
    subject_config,
) -> None:
    root = subject_config.subject_output
    source = Path(orchestrator.__file__)
    old_runner = StageRunner(
        StageContext(
            subject_config,
            Path(__file__).parents[1],
            root,
            orchestrator._base_software_provenance(),
        )
    )
    audit_value = audit_inputs(subject_config)
    old_runner.run(
        StageSpec(
            "00_input_audit",
            lambda work: (work / "input_audit.json").write_text(
                __import__("json").dumps(audit_value.to_dict()),
                encoding="utf-8",
            ),
            lambda work: (work / "input_audit.json",),
            (
                subject_config.dwi_pa,
                subject_config.bvals,
                subject_config.bvecs,
                subject_config.b0_ap,
            ),
            (source,),
        )
    )
    old_runner.run(
        StageSpec(
            "00_pre_denoise_motion_qc",
            lambda work: (work / "stripe_decision.json").write_text(
                __import__("json").dumps(
                    {
                        "subject_id": subject_config.subject_id,
                        "decision": "HOLD_FOR_REVIEW",
                        "ambiguous_reviewed": False,
                    }
                ),
                encoding="utf-8",
            ),
            lambda work: (work / "stripe_decision.json",),
            (root / "00_input_audit" / "input_audit.json",),
            (source,),
        )
    )
    reviewed = replace(
        subject_config,
        analysis=replace(subject_config.analysis, ambiguous_qc_reviewed=True),
    )
    reviewed_runner = StageRunner(
        StageContext(
            reviewed,
            Path(__file__).parents[1],
            root,
            orchestrator._base_software_provenance(),
        )
    )

    assert orchestrator._safe_manual_review_transition(
        reviewed, reviewed_runner, audit_inputs(reviewed)
    )


def _save_image(path: Path, values: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values), affine), path)


def _write_all(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _populate_valid_topup(work: Path, subject_config) -> None:
    audit = audit_inputs(subject_config)
    affine = np.asarray(audit.pa_affine)
    pa = np.stack(
        [
            np.full(audit.pa_shape[:3], 10.0 + index, dtype=np.float32)
            for index in range(len(audit.b0_indices))
        ],
        axis=3,
    )
    ap = np.stack(
        [
            np.full(audit.pa_shape[:3], 20.0 + index, dtype=np.float32)
            for index in range(audit.ap_b0_count)
        ],
        axis=3,
    )
    merged = np.concatenate((pa, ap), axis=3)
    for name, values in (
        ("nodif_PA_all.nii.gz", pa),
        ("nodif_AP_all.nii.gz", ap),
        ("PA_AP_b0.nii.gz", merged),
        ("topup_corrected_b0s.nii.gz", merged + 0.5),
    ):
        _save_image(work / name, values, affine)
    _save_image(
        work / "topup_PA_AP_b0_fieldcoef.nii.gz",
        np.zeros((*audit.pa_shape[:3], 2), dtype=np.float32),
        affine,
    )
    _save_image(
        work / "topup_field_Hz.nii.gz",
        np.zeros(audit.pa_shape[:3], dtype=np.float32),
        affine,
    )
    combined = merged.shape[3]
    np.savetxt(work / "topup_PA_AP_b0_movpar.txt", np.zeros((combined, 6)))
    pa_row = [
        *subject_config.acquisition.pa_vector,
        subject_config.acquisition.total_readout_time,
    ]
    ap_row = [
        *subject_config.acquisition.ap_vector,
        subject_config.acquisition.total_readout_time,
    ]
    np.savetxt(
        work / "acqparams_topup.txt",
        np.asarray([pa_row] * pa.shape[3] + [ap_row] * ap.shape[3]),
    )
    np.savetxt(work / "acqparams_eddy.txt", np.asarray([pa_row, ap_row]))
    np.savetxt(work / "index_eddy.txt", np.ones((1, audit.pa_shape[3])))
    np.savetxt(work / "bvals_rounded", np.loadtxt(subject_config.bvals)[None, :])
    (work / "topup_input_manifest.json").write_text(
        json.dumps(
            {
                "pa_b0_count": len(audit.b0_indices),
                "ap_b0_count": audit.ap_b0_count,
                "combined_b0_count": combined,
                "pa_b0_indices": list(audit.b0_indices),
                "volume_order": ["PA"] * len(audit.b0_indices)
                + ["AP"] * audit.ap_b0_count,
                "eddy_index_count": audit.pa_shape[3],
                "pa_acquisition_row": pa_row,
                "ap_acquisition_row": ap_row,
                "eddy_acquisition_row_order": ["PA", "AP"],
            }
        ),
        encoding="utf-8",
    )
    (work / "topup_metrics.json").write_text(
        json.dumps({"corrected_b0_count": combined}), encoding="utf-8"
    )
    _write_all(work / "topup_fsl.log")


def _realistic_quad_payload(
    quad: Path, work: Path, subject_config
) -> dict[str, object]:
    shape = audit_inputs(subject_config).pa_shape
    bvals = np.loadtxt(subject_config.bvals).reshape(-1)
    b0_count = int(np.count_nonzero(bvals < 50.0))
    dw_count = int(bvals.size - b0_count)
    outlier_percentage = 100.0 / (dw_count * shape[2])
    return {
        "eddy_input_flag": True,
        "eddy_input": {
            "field": "",
            "repol": "True",
            "topup": "topup_PA_AP_b0",
        },
        "data_file_eddy": str(work / "eddy_unwarped_images.nii.gz"),
        "data_file_mask": str(
            subject_config.subject_output / "04_bet" / "nodif_brain_mask.nii.gz"
        ),
        "data_file_bvals": str(subject_config.bvals),
        "data_no_dw_vols": dw_count,
        "data_no_b0_vols": b0_count,
        "data_no_PE_dirs": 1,
        "data_protocol": [1, 1, 1, 2, 1, 2],
        "data_no_shells": 5,
        "data_unique_bvals": [200, 500, 1000, 2000, 3000],
        "data_unique_pes": [1],
        "data_eddy_para": [0.0, -1.0, 0.0, 0.08],
        "data_vox_size": [2.0, 2.0, 2.0],
        "qc_path": str(quad),
        "qc_mot_abs": 0.25,
        "qc_mot_rel": 0.1,
        "qc_params_flag": True,
        "qc_params_avg": [0.0] * 9,
        "qc_s2v_params_flag": False,
        "qc_s2v_params_avg_std": [-1.0] * 6,
        "qc_field_flag": True,
        "qc_vox_displ_std": 0.2,
        "qc_ol_flag": True,
        "qc_outliers_tot": outlier_percentage,
        "qc_outliers_b": [0.0, 12.5, 0.0, 0.0, 0.0],
        "qc_outliers_pe": [100.0 / (shape[3] * shape[2])],
        "qc_cnr_flag": True,
        "qc_cnr_avg": [1.0] * 6,
        "qc_cnr_std": [0.1] * 6,
        "qc_rss_flag": True,
    }


def _populate_valid_quad(
    quad: Path,
    work: Path,
    subject_config,
    *,
    legacy_volumes: tuple[int, ...] | None = None,
) -> dict[str, object]:
    quad.mkdir()
    payload = _realistic_quad_payload(quad, work, subject_config)
    (quad / "qc.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    (quad / "qc.pdf").write_bytes(_minimal_pdf_bytes())
    _write_quad_msr(
        quad,
        int(payload["data_no_b0_vols"]) + int(payload["data_no_dw_vols"]),
    )
    if legacy_volumes is not None:
        (quad / "vols_no_outliers.txt").write_text(
            " ".join(str(index) for index in legacy_volumes) + "\n",
            encoding="utf-8",
        )
    return payload


def _write_quad_msr(quad: Path, volume_count: int) -> None:
    (quad / "eddy_msr.txt").write_text(
        " ".join("0.000000" for _ in range(volume_count)) + "\n",
        encoding="ascii",
    )


def _minimal_pdf_bytes() -> bytes:
    header = b"%PDF-1.4\n"
    objects = (
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>\nendobj\n",
    )
    body = bytearray(header)
    offsets = [0]
    for item in objects:
        offsets.append(len(body))
        body.extend(item)
    xref = len(body)
    body.extend(b"xref\n0 4\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        + f"startxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(body)


def _expected_sanitized_quad_metrics(
    payload: dict[str, object],
) -> dict[str, float]:
    metrics = {
        "qc_mot_abs": 0.25,
        "qc_mot_rel": 0.1,
        "qc_outliers_tot": float(payload["qc_outliers_tot"]),
        "qc_outliers_pe[0]": 1.5625,
        "qc_vox_displ_std": 0.2,
    }
    for index in range(9):
        metrics[f"qc_params_avg[{index}]"] = 0.0
    for index, value in enumerate((0.0, 12.5, 0.0, 0.0, 0.0)):
        metrics[f"qc_outliers_b[{index}]"] = value
    for index in range(6):
        metrics[f"qc_cnr_avg[{index}]"] = 1.0
        metrics[f"qc_cnr_std[{index}]"] = 0.1
    return metrics


def _populate_valid_eddy_core(work: Path, subject_config) -> None:
    audit = audit_inputs(subject_config)
    affine = np.asarray(audit.pa_affine)
    shape = audit.pa_shape
    prefix = work / "eddy_unwarped_images"
    _save_image(
        Path(f"{prefix}.nii.gz"),
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        affine,
    )
    _save_image(
        Path(f"{prefix}.eddy_residuals.nii.gz"),
        np.zeros(shape, dtype=np.float32),
        affine,
    )
    _save_image(
        Path(f"{prefix}.eddy_cnr_maps.nii.gz"),
        np.zeros((*shape[:3], 4), dtype=np.float32),
        affine,
    )
    np.savetxt(
        Path(f"{prefix}.eddy_rotated_bvecs"),
        np.loadtxt(subject_config.bvecs),
        fmt="%.8f",
    )
    np.savetxt(Path(f"{prefix}.eddy_parameters"), np.zeros((shape[3], 6)))
    np.savetxt(Path(f"{prefix}.eddy_movement_rms"), np.zeros((shape[3], 2)))
    np.savetxt(
        Path(f"{prefix}.eddy_restricted_movement_rms"),
        np.zeros((shape[3], 2)),
    )
    outliers = np.zeros((shape[3], shape[2]), dtype=int)
    outliers[2, 3] = 1
    with Path(f"{prefix}.eddy_outlier_map").open("w", encoding="utf-8") as handle:
        handle.write("Slice " + " ".join(map(str, range(shape[2]))) + "\n")
        np.savetxt(handle, outliers, fmt="%d")
    _write_all(
        Path(f"{prefix}.eddy_outlier_report"),
        "Slice 3 in scan 2 is an outlier because synthetic test\n",
    )
    _write_all(work / "eddy_fsl.log")


def _populate_valid_eddy(work: Path, subject_config) -> None:
    _populate_valid_eddy_core(work, subject_config)
    quad = work / "eddy_quad"
    payload = _populate_valid_quad(quad, work, subject_config)
    (work / "eddy_quad.json").write_text(
        json.dumps(
            {
                "metrics": _expected_sanitized_quad_metrics(payload),
                "provenance": {
                    "vols_no_outliers": "not-emitted-by-eddy-quad"
                },
            }
        ),
        encoding="utf-8",
    )


def test_successful_noddi_outputs_validate_and_promote_with_real_log_name(
    subject_config,
) -> None:
    root = subject_config.subject_output
    root.mkdir(parents=True)
    runner = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            {"python": "test"},
        )
    )

    def action(work: Path) -> None:
        names = (
            "NODDI_odi.nii",
            "NODDI_ficvf.nii",
            "NODDI_fiso.nii",
            "NODDI_kappa.nii",
            "NODDI_fmin.nii",
            "NODDI_error_code.nii",
            "NODDI_fibredirs_xvec.nii",
            "NODDI_fibredirs_yvec.nii",
            "NODDI_fibredirs_zvec.nii",
            "NODDI_params.mat",
            "noddi_metrics.json",
            "noddi_prepare.json",
            "noddi_prepare.log",
            "merge_noddi.log",
        )
        for name in names:
            _write_all(work / name, "{}\n" if name.endswith(".json") else "ok\n")

    outcome = runner.run(
        StageSpec(
            "08_noddi",
            action,
            _validate_noddi_outputs,
            (subject_config.bvals,),
            (Path(orchestrator.__file__),),
        )
    )

    assert outcome.status == "completed"
    assert (outcome.directory / "noddi_prepare.log").is_file()
    assert not (outcome.directory / "prepare_noddi.log").exists()


def test_topup_validator_checks_finite_expected_grid_and_pa_ap_order(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "topup"
    work.mkdir()
    _populate_valid_topup(work, subject_config)

    outputs = _validate_topup_outputs(
        work, audit_inputs(subject_config), subject_config
    )
    assert work / "topup_corrected_b0s.nii.gz" in outputs

    merged_path = work / "PA_AP_b0.nii.gz"
    image = nib.load(merged_path)
    reversed_data = np.asarray(image.dataobj)[..., ::-1]
    nib.save(nib.Nifti1Image(reversed_data, image.affine), merged_path)
    with pytest.raises(StageStateError, match="order|concatenation"):
        _validate_topup_outputs(work, audit_inputs(subject_config), subject_config)


def test_topup_validator_rejects_reordered_acquisition_rows(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "topup"
    work.mkdir()
    _populate_valid_topup(work, subject_config)
    rows = np.loadtxt(work / "acqparams_topup.txt")
    np.savetxt(work / "acqparams_topup.txt", rows[::-1])

    with pytest.raises(StageStateError, match="acquisition.*order|PA/AP"):
        _validate_topup_outputs(
            work, audit_inputs(subject_config), subject_config
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dwi_shape", "shape|grid"),
        ("dwi_nonfinite", "finite"),
        ("bvec_shape", "3xN|3"),
        ("bvec_norm", "unit|norm"),
        ("outlier_nonbinary", "0/1|binary|integral"),
        ("outlier_report", "report|outlier"),
        ("residual_shape", "residual|shape|grid"),
        ("cnr_nonfinite", "CNR|finite"),
    ],
)
def test_eddy_validator_rejects_malformed_scientific_outputs_at_eddy_stage(
    subject_config, tmp_path: Path, mutation: str, message: str
) -> None:
    work = tmp_path / mutation
    work.mkdir()
    _populate_valid_eddy(work, subject_config)
    prefix = work / "eddy_unwarped_images"
    affine = np.asarray(audit_inputs(subject_config).pa_affine)
    if mutation == "dwi_shape":
        _save_image(Path(f"{prefix}.nii.gz"), np.zeros((8, 8, 8, 7)), affine)
    elif mutation == "dwi_nonfinite":
        values = np.zeros((8, 8, 8, 8), dtype=np.float32)
        values[0, 0, 0, 0] = np.nan
        _save_image(Path(f"{prefix}.nii.gz"), values, affine)
    elif mutation == "bvec_shape":
        np.savetxt(Path(f"{prefix}.eddy_rotated_bvecs"), np.zeros((8, 3)))
    elif mutation == "bvec_norm":
        values = np.loadtxt(subject_config.bvecs)
        values[:, 1] *= 0.5
        np.savetxt(Path(f"{prefix}.eddy_rotated_bvecs"), values)
    elif mutation == "outlier_nonbinary":
        outlier = np.zeros((8, 8))
        outlier[0, 0] = 2
        with Path(f"{prefix}.eddy_outlier_map").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write("Slice " + " ".join(map(str, range(8))) + "\n")
            np.savetxt(handle, outlier)
    elif mutation == "outlier_report":
        _write_all(Path(f"{prefix}.eddy_outlier_report"), "unsupported\n")
    elif mutation == "residual_shape":
        _save_image(
            Path(f"{prefix}.eddy_residuals.nii.gz"),
            np.zeros((7, 8, 8, 8)),
            affine,
        )
    elif mutation == "cnr_nonfinite":
        values = np.zeros((8, 8, 8, 2), dtype=np.float32)
        values[0, 0, 0, 0] = np.inf
        _save_image(Path(f"{prefix}.eddy_cnr_maps.nii.gz"), values, affine)

    with pytest.raises(StageStateError, match=message):
        _validate_eddy_outputs(
            work,
            audit_inputs(subject_config),
            subject_config.bvals,
        )


def test_large_4d_output_finiteness_validation_streams_one_volume_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed: list[int] = []

    class Proxy:
        def __array__(self, *args, **kwargs):
            raise AssertionError("validator attempted to materialize the full 4D image")

        def __getitem__(self, key):
            accessed.append(int(key[-1]))
            return np.zeros((2, 2, 2), dtype=np.float32)

    image = type(
        "StreamingImage",
        (),
        {"shape": (2, 2, 2, 3), "affine": np.eye(4), "dataobj": Proxy()},
    )()
    monkeypatch.setattr(orchestrator.nib, "load", lambda path: image)

    orchestrator._validate_finite_nifti(
        tmp_path / "large.nii.gz",
        "large synthetic output",
        (2, 2, 2, 3),
        np.eye(4),
    )

    assert accessed == [0, 1, 2]


def test_topup_concatenation_validation_streams_each_input_volume() -> None:
    accessed: list[tuple[str, int]] = []

    class Proxy:
        def __init__(self, name: str, values: tuple[float, ...]) -> None:
            self.name = name
            self.values = values

        def __array__(self, *args, **kwargs):
            raise AssertionError("validator attempted to materialize a full 4D image")

        def __getitem__(self, key):
            volume = int(key[-1])
            accessed.append((self.name, volume))
            return np.full((2, 2, 2), self.values[volume], dtype=np.float32)

    def image(name: str, values: tuple[float, ...]):
        return type(
            f"{name.title()}Image",
            (),
            {
                "shape": (2, 2, 2, len(values)),
                "dataobj": Proxy(name, values),
            },
        )()

    orchestrator._validate_concatenated_nifti_volumes(
        image("pa", (1.0, 2.0)),
        image("ap", (3.0,)),
        image("merged", (1.0, 2.0, 3.0)),
        "TOPUP merged b0 volume order",
    )

    assert accessed == [
        ("pa", 0),
        ("merged", 0),
        ("pa", 1),
        ("merged", 1),
        ("ap", 0),
        ("merged", 2),
    ]


def test_eddy_validator_accepts_complete_finite_outputs_and_quad_source(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    _populate_valid_eddy(work, subject_config)

    outputs = _validate_eddy_outputs(
        work, audit_inputs(subject_config), subject_config.bvals
    )

    assert work / "eddy_quad" / "qc.json" in outputs
    assert work / "eddy_quad.json" in outputs


def test_eddy_quad_source_tree_is_validated_before_json_is_derived(
    tmp_path: Path,
) -> None:
    source = tmp_path / "eddy_quad"
    source.mkdir()
    external = tmp_path / "external.json"
    external.write_text('{"qc_motion": 1.0}\n', encoding="utf-8")
    (source / "unsafe.json").symlink_to(external)
    destination = tmp_path / "eddy_quad.json"

    with pytest.raises(StageStateError, match="symbolic link"):
        _write_eddy_quad_json(source, destination)

    assert not destination.exists()


def test_eddy_quad_rejects_arbitrary_numeric_json_without_real_products(
    tmp_path: Path,
) -> None:
    source = tmp_path / "eddy_quad"
    source.mkdir()
    (source / "foo.json").write_text('{"unrelated": 1}\n', encoding="utf-8")

    with pytest.raises(StageStateError, match="qc.json|qc.pdf"):
        _write_eddy_quad_json(source, tmp_path / "eddy_quad.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_pdf", "qc.pdf"),
        ("empty_pdf", "qc.pdf"),
        ("malformed_pdf", "qc.pdf|PDF"),
        ("missing_msr", "eddy_msr"),
        ("malformed_msr", "eddy_msr"),
        ("nonfinite_msr", "eddy_msr|finite"),
        ("wrong_msr_count", "eddy_msr|volume"),
        ("duplicate_volumes", "vols_no_outliers"),
        ("duplicate_json_key", "duplicate"),
        ("missing_json_key", "schema|required"),
        ("unexpected_json_key", "schema|unexpected"),
        ("nonfinite_json_number", "finite"),
        ("inconsistent_volume_count", "volume"),
        ("unsafe_qc_path", "qc_path|path"),
        ("hardlinked_pdf", "single link|hard link"),
    ),
)
def test_eddy_quad_rejects_malformed_required_contract(
    subject_config,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    payload = _populate_valid_quad(quad, work, subject_config)
    if mutation == "missing_pdf":
        (quad / "qc.pdf").unlink()
    elif mutation == "empty_pdf":
        (quad / "qc.pdf").write_bytes(b"")
    elif mutation == "malformed_pdf":
        (quad / "qc.pdf").write_bytes(b"%PDF-1.4\ntrailer\n%%EOF\n")
    elif mutation == "missing_msr":
        (quad / "eddy_msr.txt").unlink()
    elif mutation == "malformed_msr":
        (quad / "eddy_msr.txt").write_text(
            "0.000000\n0.000000\n",
            encoding="ascii",
        )
    elif mutation == "nonfinite_msr":
        (quad / "eddy_msr.txt").write_text(
            "nan " + " ".join("0.000000" for _ in range(7)) + "\n",
            encoding="ascii",
        )
    elif mutation == "wrong_msr_count":
        _write_quad_msr(quad, 7)
    elif mutation == "duplicate_volumes":
        (quad / "vols_no_outliers.txt").write_text("0\n0\n", encoding="utf-8")
    elif mutation == "duplicate_json_key":
        text = json.dumps(payload)
        (quad / "qc.json").write_text(
            text[:-1] + ', "qc_mot_abs": 99.0}',
            encoding="utf-8",
        )
    elif mutation == "missing_json_key":
        del payload["qc_mot_rel"]
        (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "unexpected_json_key":
        payload["unexpected"] = 1
        (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "nonfinite_json_number":
        payload["qc_mot_abs"] = float("nan")
        (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "inconsistent_volume_count":
        payload["data_no_dw_vols"] = 8
        (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "unsafe_qc_path":
        payload["qc_path"] = "../escape"
        (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "hardlinked_pdf":
        os.link(quad / "qc.pdf", tmp_path / "second-qc.pdf")
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(StageStateError, match=message):
        _write_eddy_quad_json(quad, tmp_path / "sanitized.json")


def test_eddy_quad_realistic_contract_produces_curated_finite_metrics(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    payload = _populate_valid_quad(quad, work, subject_config)
    destination = work / "eddy_quad.json"

    _write_eddy_quad_json(quad, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "metrics": _expected_sanitized_quad_metrics(payload),
        "provenance": {
            "vols_no_outliers": "not-emitted-by-eddy-quad"
        },
    }


def test_current_quad_absence_does_not_create_a_legacy_volume_artifact(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    payload = _populate_valid_quad(quad, work, subject_config)
    destination = work / "eddy_quad.json"

    _write_eddy_quad_json(quad, destination)

    assert not (quad / "vols_no_outliers.txt").exists()
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "metrics": _expected_sanitized_quad_metrics(payload),
        "provenance": {
            "vols_no_outliers": "not-emitted-by-eddy-quad"
        },
    }


def test_eddy_validator_recomputes_quad_provenance_from_the_promoted_tree(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    _populate_valid_eddy(work, subject_config)
    sanitized = work / "eddy_quad.json"
    payload = json.loads(sanitized.read_text(encoding="utf-8"))
    payload["provenance"]["vols_no_outliers"] = (
        "eddy-quad-legacy-residual-msr"
    )
    sanitized.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageStateError, match="provenance|vols_no_outliers"):
        _validate_eddy_outputs(
            work,
            audit_inputs(subject_config),
            subject_config.bvals,
        )


@pytest.mark.parametrize(
    ("raw_bvals", "b_range", "unique_bvals", "protocol"),
    (
        ((0.0, 995.0, 995.0), None, (995,), (1, 2)),
        (
            (0.0, 180.0, 190.0, 200.0, 980.0, 990.0, 1000.0),
            None,
            (190, 990),
            (1, 3, 3),
        ),
        ((0.0, 400.0, 490.0, 580.0), None, (490,), (1, 3)),
        ((0.0, 40.0, 900.0, 980.0), "50", (900, 980), (2, 1, 1)),
    ),
)
def test_current_quad_accepts_connected_component_median_shells(
    subject_config,
    tmp_path: Path,
    raw_bvals: tuple[float, ...],
    b_range: str | None,
    unique_bvals: tuple[int, ...],
    protocol: tuple[int, ...],
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    payload = _populate_valid_quad(quad, work, subject_config)
    if b_range is not None:
        payload["eddy_input"]["b_range"] = b_range
    payload["data_no_b0_vols"] = protocol[0]
    payload["data_no_dw_vols"] = len(raw_bvals) - protocol[0]
    payload["data_no_shells"] = len(unique_bvals)
    payload["data_unique_bvals"] = list(unique_bvals)
    payload["data_protocol"] = list(protocol)
    payload["qc_outliers_tot"] = 0.0
    payload["qc_outliers_b"] = [0.0] * len(unique_bvals)
    payload["qc_outliers_pe"] = [0.0]
    payload["qc_cnr_avg"] = [1.0] * (len(unique_bvals) + 1)
    payload["qc_cnr_std"] = [0.1] * (len(unique_bvals) + 1)
    (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_quad_msr(quad, len(raw_bvals))
    destination = work / "eddy_quad.json"

    _write_eddy_quad_json(
        quad,
        destination,
        expected_volume_count=len(raw_bvals),
        expected_slices=8,
        bvals=np.asarray(raw_bvals),
        outliers=np.zeros((len(raw_bvals), 8), dtype=np.uint8),
    )

    sanitized = json.loads(destination.read_text(encoding="utf-8"))
    assert sanitized["provenance"] == {
        "vols_no_outliers": "not-emitted-by-eddy-quad"
    }
    assert [
        sanitized["metrics"][f"qc_outliers_b[{index}]"]
        for index in range(len(unique_bvals))
    ] == [0.0] * len(unique_bvals)


def test_legacy_residual_volume_list_may_differ_from_slice_outlier_complement(
    subject_config, tmp_path: Path
) -> None:
    raw_bvals = np.asarray([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    payload = _populate_valid_quad(
        quad,
        work,
        subject_config,
        legacy_volumes=(0, 1, 2, 3, 4, 5),
    )
    payload["data_no_b0_vols"] = 1
    payload["data_no_dw_vols"] = 6
    payload["data_no_shells"] = 1
    payload["data_unique_bvals"] = [1000]
    payload["data_protocol"] = [1, 6]
    payload["qc_outliers_tot"] = 100.0 / 48.0
    payload["qc_outliers_b"] = [100.0 / 48.0]
    payload["qc_outliers_pe"] = [100.0 / 56.0]
    payload["qc_cnr_avg"] = [1.0, 1.0]
    payload["qc_cnr_std"] = [0.1, 0.1]
    (quad / "qc.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_quad_msr(quad, 7)
    outliers = np.zeros((7, 8), dtype=np.uint8)
    outliers[2, 3] = 1
    destination = work / "eddy_quad.json"

    _write_eddy_quad_json(
        quad,
        destination,
        expected_volume_count=7,
        expected_slices=8,
        bvals=raw_bvals,
        outliers=outliers,
    )

    assert tuple(np.flatnonzero(~np.any(outliers.astype(bool), axis=1))) == (
        0,
        1,
        3,
        4,
        5,
        6,
    )
    assert json.loads(destination.read_text(encoding="utf-8"))[
        "provenance"
    ] == {"vols_no_outliers": "eddy-quad-legacy-residual-msr"}


@pytest.mark.parametrize(
    "legacy_text",
    ("0  1\n", "00 1\n", "0 1", "0\t1\n"),
)
def test_legacy_volume_list_requires_canonical_producer_syntax(
    subject_config,
    tmp_path: Path,
    legacy_text: str,
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    _populate_valid_quad(quad, work, subject_config)
    (quad / "vols_no_outliers.txt").write_text(
        legacy_text,
        encoding="ascii",
    )

    with pytest.raises(StageStateError, match="vols_no_outliers|malformed"):
        _write_eddy_quad_json(quad, work / "eddy_quad.json")


def test_legacy_volume_list_rejects_an_unbounded_integer_token_as_stage_state(
    subject_config,
    tmp_path: Path,
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    quad = work / "eddy_quad"
    _populate_valid_quad(quad, work, subject_config)
    (quad / "vols_no_outliers.txt").write_text(
        "9" * 5000 + "\n",
        encoding="ascii",
    )

    with pytest.raises(StageStateError, match="vols_no_outliers|bounded"):
        _write_eddy_quad_json(quad, work / "eddy_quad.json")


def test_eddy_quad_rejects_outlier_percentages_inconsistent_with_eddy_map(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    _populate_valid_eddy(work, subject_config)
    qc_json = work / "eddy_quad" / "qc.json"
    payload = json.loads(qc_json.read_text(encoding="utf-8"))
    payload["qc_outliers_tot"] = 99.0
    qc_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageStateError, match="outlier|percentage|inconsistent"):
        _validate_eddy_outputs(
            work,
            audit_inputs(subject_config),
            subject_config.bvals,
        )


def test_real_eddy_stage_validator_rejects_wrong_quad_mask_path(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    _populate_valid_eddy(work, subject_config)
    qc_json = work / "eddy_quad" / "qc.json"
    payload = json.loads(qc_json.read_text(encoding="utf-8"))
    payload["data_file_mask"] = str(tmp_path / "wrong-mask.nii.gz")
    qc_json.write_text(json.dumps(payload), encoding="utf-8")
    eddy = {
        spec.name: spec for spec in build_plan(subject_config)
    }["05_eddy"]

    with pytest.raises(StageStateError, match="data_file_mask|path"):
        eddy.validator(work)


def test_eddy_validator_accepts_real_quadratic_model_parameter_width(
    subject_config, tmp_path: Path
) -> None:
    work = tmp_path / "eddy"
    work.mkdir()
    _populate_valid_eddy(work, subject_config)
    prefix = work / "eddy_unwarped_images"
    volumes = audit_inputs(subject_config).pa_shape[3]
    np.savetxt(
        Path(f"{prefix}.eddy_parameters"),
        np.zeros((volumes, 16), dtype=np.float64),
    )

    _validate_eddy_outputs(
        work, audit_inputs(subject_config), subject_config.bvals
    )


def _fake_runtime(tmp_path: Path, subject_config) -> tuple[orchestrator._Runtime, list[Path]]:
    fsldir = tmp_path / "fsl"
    names = (
        "topup",
        "applytopup",
        "bet",
        "fslmaths",
        "eddy",
        "eddy_quad",
        "flirt",
        "fnirt",
        "invwarp",
        "applywarp",
    )
    tools: dict[str, Path] = {}
    for name in names:
        path = fsldir / "bin" / name
        _write_all(path, f"{name}\n")
        path.chmod(0o755)
        tools[name] = path
    _write_all(
        tools["eddy"],
        "#!/usr/bin/env fslpython\nfrom fsl.base import find_cuda_exe\n",
    )
    _write_all(
        tools["eddy_quad"],
        (
            "#!/bin/sh\n"
            "'''exec' \"${FSLDIR}/bin/python\" \"$0\" \"$@\"\n"
            "' '''\n"
            "from eddy_qc.scripts.eddy_quad import main\n"
        ),
    )
    for relative in (
        "bin/bet2",
        "bin/eddy_cpu",
        "bin/find_cuda_exe",
        "bin/remove_ext",
        "bin/imtest",
        "bin/imrm",
        "bin/imglob",
        "bin/fslval",
        "bin/fslhd",
        "bin/fslstats",
        "bin/fslsplit",
        "bin/python3.12",
        "bin/slicer",
    ):
        path = fsldir / relative
        _write_all(path, relative + "\n")
        path.chmod(0o755)
    _write_all(
        fsldir / "bin" / "fslpython",
        '#!/bin/sh\nexec "${FSLDIR}/bin/python" "$@"\n',
    )
    (fsldir / "bin" / "fslpython").chmod(0o755)
    (fsldir / "bin" / "python").symlink_to("python3.12")
    for relative in (
        "lib/python3.12/site-packages/eddy_qc/__init__.py",
        "lib/python3.12/site-packages/eddy_qc/scripts/eddy_quad.py",
        "lib/python3.12/site-packages/eddy_qc/QUAD/quad.py",
        "lib/python3.12/site-packages/fsl/base/__init__.py",
        "lib/python3.12/site-packages/fsl/base/find_cuda_exe.py",
        "lib/python3.12/site-packages/fsl/data/__init__.py",
        "lib/python3.12/site-packages/fsl/data/image.py",
        "lib/python3.12/site-packages/fsl/scripts/remove_ext.py",
        "lib/python3.12/site-packages/fsl/scripts/imtest.py",
        "lib/python3.12/site-packages/fsl/transform/__init__.py",
        "lib/python3.12/site-packages/fsl/transform/affine.py",
        "lib/python3.12/site-packages/fsl/utils/path.py",
        "lib/python3.12/site-packages/fsl/utils/run.py",
        "lib/python3.12/site-packages/fsl/wrappers/__init__.py",
        "lib/python3.12/site-packages/numpy/__init__.py",
        "lib/python3.12/site-packages/numpy/_core/fromnumeric.py",
        "lib/python3.12/site-packages/nibabel/__init__.py",
        "lib/python3.12/site-packages/nibabel/nifti1.py",
        "lib/python3.12/site-packages/matplotlib/__init__.py",
        "lib/python3.12/site-packages/matplotlib/backends/backend_pdf.py",
        "lib/python3.12/site-packages/matplotlib/mpl-data/matplotlibrc",
        "lib/python3.12/site-packages/seaborn/__init__.py",
        "lib/python3.12/site-packages/seaborn/categorical.py",
    ):
        _write_all(fsldir / relative, relative + "\n")
    _write_all(
        fsldir
        / "lib/python3.12/site-packages/eddy_qc-1.4.2.dist-info/METADATA",
        "Metadata-Version: 2.1\nName: eddy_qc\nVersion: 1.4.2\n",
    )
    for distribution, version in (
        ("numpy", "2.1.3"),
        ("nibabel", "5.4.2"),
        ("matplotlib", "3.9.4"),
        ("seaborn", "0.13.2"),
    ):
        metadata_root = (
            fsldir
            / "lib/python3.12/site-packages"
            / f"{distribution}-{version}.dist-info"
        )
        _write_all(
            metadata_root / "METADATA",
            (
                "Metadata-Version: 2.1\n"
                f"Name: {distribution}\n"
                f"Version: {version}\n"
            ),
        )
        _write_all(metadata_root / "RECORD", f"{distribution}\n")
    resources = []
    for relative in (
        "etc/flirtsch/b02b0.cnf",
        "etc/flirtsch/b02b0_1.cnf",
        "etc/flirtsch/FA_2_FMRIB58_1mm.cnf",
        "data/standard/FMRIB58_FA_1mm.nii.gz",
    ):
        path = fsldir / relative
        _write_all(path, relative + "\n")
        resources.append(path)
    matlab = tmp_path / "MATLAB.app" / "bin" / "matlab"
    _write_all(matlab, "matlab\n")
    matlab.chmod(0o755)
    installation = FSLInstallation(
        fsldir=fsldir,
        topup=tools["topup"],
        applytopup=tools["applytopup"],
        bet=tools["bet"],
        fslmaths=tools["fslmaths"],
        eddy=tools["eddy"],
        eddy_quad=tools["eddy_quad"],
        flirt=tools["flirt"],
        fnirt=tools["fnirt"],
        invwarp=tools["invwarp"],
        applywarp=tools["applywarp"],
        b02b0_config=resources[0],
        b02b0_no_subsampling_config=resources[1],
        fa_to_standard_config=resources[2],
        standard_fa=resources[3],
        _environment_items=(("FSLDIR", str(fsldir)),),
    )
    runtime = orchestrator._Runtime(
        subject_config,
        fsl=installation,
        matlab=MATLABInstallation(
            executable=matlab,
            version="25.1",
            mexext="mexmaca64",
            optimization_toolbox=True,
            mex_configured=True,
        ),
    )
    return runtime, [*tools.values(), *resources, matlab]


def test_software_provenance_tracks_every_material_fsl_and_matlab_file(
    subject_config, tmp_path: Path
) -> None:
    runtime, files = _fake_runtime(tmp_path, subject_config)
    before = dict(_software_provenance(runtime))

    expected_keys = (
        "fsl_topup_sha256",
        "fsl_applytopup_sha256",
        "fsl_bet_sha256",
        "fsl_fslmaths_sha256",
        "fsl_eddy_sha256",
        "fsl_eddy_quad_sha256",
        "fsl_flirt_sha256",
        "fsl_fnirt_sha256",
        "fsl_invwarp_sha256",
        "fsl_applywarp_sha256",
        "fsl_b02b0_config_sha256",
        "fsl_b02b0_no_subsampling_config_sha256",
        "fsl_fa_to_standard_config_sha256",
        "fsl_standard_fa_sha256",
        "matlab_executable_sha256",
    )
    assert set(expected_keys).issubset(before)
    for path, key in zip(files, expected_keys, strict=True):
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        after = dict(_software_provenance(runtime))
        assert after[key] != before[key], path
        path.write_bytes(original)


def test_software_provenance_rejects_symlinked_material_dependency(
    subject_config, tmp_path: Path
) -> None:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    replacement = tmp_path / "replacement-tool"
    replacement.write_text("replacement\n", encoding="utf-8")
    runtime.fsl.bet.unlink()
    runtime.fsl.bet.symlink_to(replacement)

    with pytest.raises(
        orchestrator.PipelineDependencyError,
        match="explicit regular file",
    ):
        _software_provenance(runtime)


def test_material_tool_replacement_makes_scientific_stage_noncurrent(
    subject_config, tmp_path: Path
) -> None:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    root = subject_config.subject_output
    root.mkdir(parents=True)
    source = Path(orchestrator.__file__)
    spec = StageSpec(
        "05_eddy",
        lambda work: (work / "result.txt").write_text(
            "valid\n", encoding="utf-8"
        ),
        lambda work: (work / "result.txt",),
        (subject_config.bvals,),
        (source,),
    )
    first = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            _software_provenance(runtime),
        )
    )
    first.run(spec)
    runtime.fsl.bet.write_text("materially changed\n", encoding="utf-8")
    second = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            _software_provenance(runtime),
        )
    )

    assert not second.is_current(spec)


@pytest.mark.parametrize(
    "relative",
    (
        "bin/bet2",
        "bin/fslpython",
        "bin/fslstats",
        "bin/eddy_cpu",
        "lib/python3.12/site-packages/eddy_qc/QUAD/quad.py",
        "lib/python3.12/site-packages/fsl/data/image.py",
        "lib/python3.12/site-packages/fsl/wrappers/__init__.py",
        "lib/python3.12/site-packages/numpy/_core/fromnumeric.py",
        "lib/python3.12/site-packages/nibabel/nifti1.py",
        "lib/python3.12/site-packages/matplotlib/mpl-data/matplotlibrc",
        "lib/python3.12/site-packages/seaborn/categorical.py",
        "lib/python3.12/site-packages/numpy-2.1.3.dist-info/METADATA",
        "lib/python3.12/site-packages/nibabel-5.4.2.dist-info/METADATA",
        "lib/python3.12/site-packages/matplotlib-3.9.4.dist-info/METADATA",
        "lib/python3.12/site-packages/seaborn-0.13.2.dist-info/METADATA",
    ),
)
def test_runtime_backend_replacement_makes_stage_noncurrent_without_path_leak(
    subject_config, tmp_path: Path, relative: str
) -> None:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    material = []
    for name in (
        "bin/bet2",
        "bin/fslpython",
        "bin/fslstats",
        "bin/eddy_cpu",
        "lib/python3.12/site-packages/eddy_qc/QUAD/quad.py",
        "lib/python3.12/site-packages/fsl/data/image.py",
        "lib/python3.12/site-packages/fsl/wrappers/__init__.py",
        "lib/python3.12/site-packages/numpy/_core/fromnumeric.py",
        "lib/python3.12/site-packages/nibabel/nifti1.py",
        "lib/python3.12/site-packages/matplotlib/mpl-data/matplotlibrc",
        "lib/python3.12/site-packages/seaborn/categorical.py",
        "lib/python3.12/site-packages/numpy-2.1.3.dist-info/METADATA",
        "lib/python3.12/site-packages/nibabel-5.4.2.dist-info/METADATA",
        "lib/python3.12/site-packages/matplotlib-3.9.4.dist-info/METADATA",
        "lib/python3.12/site-packages/seaborn-0.13.2.dist-info/METADATA",
    ):
        path = runtime.fsl.fsldir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        material.append((name, path))
    object.__setattr__(
        runtime.fsl,
        "_runtime_material_items",
        tuple(material),
    )
    software = _software_provenance(runtime)
    serialized = json.dumps(dict(software), sort_keys=True)
    assert str(tmp_path) not in serialized

    root = subject_config.subject_output
    root.mkdir(parents=True)
    source = Path(orchestrator.__file__)
    spec = StageSpec(
        "05_eddy",
        lambda work: (work / "result.txt").write_text("valid\n", encoding="utf-8"),
        lambda work: (work / "result.txt",),
        (subject_config.bvals,),
        (source,),
    )
    first = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            software,
        )
    )
    first.run(spec)
    (runtime.fsl.fsldir / relative).write_text("replacement\n", encoding="utf-8")
    second = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            _software_provenance(runtime),
        )
    )

    assert not second.is_current(spec)


def test_force_rejects_raw_input_inside_subject_output_before_mutation(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = subject_config.subject_output
    nested = root / "05_eddy" / "raw_pa.nii.gz"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(subject_config.dwi_pa.read_bytes())
    unsafe = replace(subject_config, dwi_pa=nested)
    original = nested.read_bytes()
    monkeypatch.setattr(
        orchestrator,
        "_build_plan",
        lambda config, runtime: pytest.fail("plan built before path separation check"),
    )

    with pytest.raises(
        orchestrator.PipelineInputError,
        match="input|subject output|inside",
    ):
        run_pipeline(unsafe, "run", force_stage="05_eddy")

    assert nested.is_file()
    assert nested.read_bytes() == original
    assert not (root / ".invalidated").exists()


def test_subject_output_cannot_be_nested_lexically_below_a_raw_file(
    subject_config,
) -> None:
    unsafe = replace(
        subject_config,
        output_root=subject_config.dwi_pa / "nested-output",
    )
    before = subject_config.dwi_pa.read_bytes()

    with pytest.raises(orchestrator.PipelineInputError, match="inside|below"):
        run_pipeline(unsafe, "run")

    assert subject_config.dwi_pa.read_bytes() == before
    assert not unsafe.subject_output.exists()


def test_report_stage_tracks_every_consumed_stripe_detail_and_qc_figure(
    subject_config,
) -> None:
    report = {spec.name: spec for spec in build_plan(subject_config)}["report"]
    root = subject_config.subject_output
    expected_details = {
        root
        / "00_pre_denoise_motion_qc"
        / "03_candidate_details_01_06.png",
        root
        / "00_pre_denoise_motion_qc"
        / "03_candidate_details_07_08.png",
        root
        / "00_pre_denoise_motion_qc"
        / "04_all_volumes_001_008.png",
    }
    expected_qc = {
        root
        / "qc"
        / name.replace("${subject_id}", subject_config.subject_id)
        for name in orchestrator.FIGURE_FILENAMES.values()
    }

    assert expected_details.issubset(report.input_paths)
    assert expected_qc.issubset(report.input_paths)


def test_preoutput_plan_tracks_more_than_twelve_union_selected_detail_volumes(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_count = 20
    inputs = tmp_path / "twenty-volume-inputs"
    inputs.mkdir()
    source = nib.load(subject_config.dwi_pa)
    volume = np.asarray(source.dataobj[..., 0], dtype=np.float32)
    dwi = np.stack(
        [volume + float(index) for index in range(volume_count)],
        axis=-1,
    )
    dwi_path = inputs / "pa_20.nii.gz"
    bvals_path = inputs / "pa_20.bval"
    bvecs_path = inputs / "pa_20.bvec"
    nib.save(nib.Nifti1Image(dwi, source.affine), dwi_path)
    bvals = np.asarray([0.0] + [1000.0] * (volume_count - 1))
    np.savetxt(bvals_path, bvals[None, :], fmt="%.0f")
    original_bvecs = np.loadtxt(subject_config.bvecs)
    bvecs = np.zeros((3, volume_count), dtype=float)
    for index in range(1, volume_count):
        bvecs[:, index] = original_bvecs[:, 1 + ((index - 1) % 7)]
    np.savetxt(bvecs_path, bvecs, fmt="%.8f")
    config = replace(
        subject_config,
        dwi_pa=dwi_path,
        bvals=bvals_path,
        bvecs=bvecs_path,
        analysis=replace(
            subject_config.analysis,
            ambiguous_qc_reviewed=True,
        ),
    )
    csi = np.asarray([1.26] * 4 + [1.20] * 9 + [1.0] * 7)
    metrics = StripeMetrics(
        a_si=csi,
        c_si=csi,
        shells=np.asarray([0] + [1000] * 19),
        peak_sagittal=np.zeros(volume_count, dtype=int),
    )
    monkeypatch.setattr(
        stripe_qc,
        "compute_stripe_indices",
        lambda image, values: metrics,
    )

    preoutput_plan = build_plan(config)
    report = {spec.name: spec for spec in preoutput_plan}["report"]
    stripe = config.subject_output / "00_pre_denoise_motion_qc"
    expected = {
        stripe / "03_candidate_details_01_06.png",
        stripe / "03_candidate_details_07_12.png",
        stripe / "03_candidate_details_13_13.png",
        stripe / "04_all_volumes_001_020.png",
    }

    assert expected.issubset(report.input_paths)

    captured_contexts = []

    def fake_report(context) -> None:
        captured_contexts.append(context)
        detail_sizes = {
            path.name: len(path.read_bytes())
            for path in context.stripe_detail_files
        }
        _write_all(
            context.output_directory / f"{config.subject_id}_QC_report.pdf",
            "synthetic PDF\n",
        )
        _write_all(
            context.output_directory
            / f"{config.subject_id}_analysis_report.md",
            "synthetic report\n",
        )
        _write_all(
            context.output_directory / f"{config.subject_id}_run_summary.json",
            json.dumps({"stripe_detail_sizes": detail_sizes}) + "\n",
        )

    monkeypatch.setattr(orchestrator, "write_final_report", fake_report)
    monkeypatch.setattr(
        orchestrator,
        "_software_provenance",
        lambda runtime: {"synthetic_runtime": "fixed"},
    )
    config.subject_output.mkdir(parents=True)
    gate_runner = StageRunner(
        StageContext(
            config,
            Path(orchestrator.__file__).parents[2],
            config.subject_output,
            orchestrator._base_software_provenance(),
        )
    )
    for spec in preoutput_plan[:2]:
        gate_runner.run(spec)
    monkeypatch.setattr(
        stripe_qc,
        "compute_stripe_indices",
        lambda image, values: (_ for _ in ()).throw(
            AssertionError("stripe-detail dependencies were recomputed")
        ),
    )
    for path in report.input_paths:
        if not path.exists():
            _write_all(path, "{}\n")
    report_runner = StageRunner(
        StageContext(
            config,
            Path(orchestrator.__file__).parents[2],
            config.subject_output,
            {"synthetic_runtime": "fixed"},
        )
    )

    first = report_runner.run(report)
    second = report_runner.run(report)

    assert first.status == "completed"
    assert second.status == "skipped"
    assert len(captured_contexts) == 1
    assert set(captured_contexts[0].stripe_detail_files) == expected
    assert all(path.is_file() for path in expected)
    assert report_runner.is_current(report)
    record = json.loads(first.record_path.read_text(encoding="utf-8"))
    recorded_inputs = {
        item["path"] for item in record["inputs"]
    }
    assert {
        path.relative_to(config.config_path.parent).as_posix()
        for path in expected
    }.issubset(recorded_inputs)
    run_summary = json.loads(
        (
            first.directory
            / f"{config.subject_id}_run_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert set(run_summary["stripe_detail_sizes"]) == {
        path.name for path in expected
    }


def test_dry_run_commands_use_real_work_paths_and_absolute_early_bet(
    subject_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    monkeypatch.setattr(orchestrator, "_installed_memory_gib", lambda: 64.0)

    commands = _dry_run_commands(
        subject_config, audit_inputs(subject_config), runtime
    )

    root = subject_config.subject_output
    assert commands[0] == [
        str(runtime.fsl.bet),
        str(root / ".work" / "01_denoise" / "raw_mean_b0.nii.gz"),
        str(root / ".work" / "01_denoise" / "raw_mean_b0_bet"),
        "-R",
        "-f",
        "0.25",
        "-g",
        "0",
        "-m",
    ]
    rendered = "\n".join("\0".join(argv) for argv in commands)
    for stage in ("03_topup", "04_bet", "05_eddy", "08_noddi", "09_jhu_48roi"):
        assert str(root / ".work" / stage) in rendered
    assert commands[2][1] == str(
        root / "03_topup" / "topup_corrected_b0s"
    )
    assert commands[2][-1] == str(root / ".work" / "04_bet" / "hifi_nodif")
    eddy_command = commands[4]
    assert f"--mask={root / '04_bet' / 'nodif_brain_mask.nii.gz'}" in eddy_command
    assert f"--topup={root / '03_topup' / 'topup_PA_AP_b0'}" in eddy_command
    assert f"--out={root / '.work' / '05_eddy' / 'eddy_unwarped_images'}" in eddy_command
    assert not root.exists()


def test_dry_run_uses_discovered_absolute_bet_and_fsl_environment(
    subject_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded, _ = _fake_runtime(tmp_path, subject_config)
    configured = replace(subject_config, fsldir=seeded.fsl.fsldir)
    runtime = orchestrator._Runtime(configured, matlab=seeded.matlab)
    monkeypatch.setattr(orchestrator, "_installed_memory_gib", lambda: 64.0)

    commands = _dry_run_commands(configured, audit_inputs(configured), runtime)

    assert commands[0][0] == str(seeded.fsl.fsldir / "bin" / "bet")
    assert Path(commands[0][0]).is_absolute()
    assert runtime.fsl.environment["FSLDIR"] == str(seeded.fsl.fsldir)
    assert runtime.fsl.environment["FSLOUTPUTTYPE"] == "NIFTI_GZ"


def test_dry_run_wraps_memory_probe_failure_as_typed_external_error(
    subject_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    monkeypatch.setattr(
        orchestrator,
        "_installed_memory_gib",
        lambda: (_ for _ in ()).throw(NODDIError("cannot read /proc/meminfo")),
    )

    with pytest.raises(orchestrator.PipelineExternalError, match="/proc/meminfo"):
        _dry_run_commands(
            subject_config, audit_inputs(subject_config), runtime
        )


def _install_real_plan_test_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    subject_config,
    tmp_path: Path,
    csi: np.ndarray,
) -> tuple[orchestrator._Runtime, list[str], list[str]]:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    action_calls: list[str] = []
    discoveries: list[str] = []

    def discover_fsl(config):
        discoveries.append("fsl")
        return runtime.fsl

    def discover_matlab(config):
        discoveries.append("matlab")
        return runtime.matlab

    monkeypatch.setattr(orchestrator, "discover_fsl", discover_fsl)
    monkeypatch.setattr(orchestrator, "discover_matlab", discover_matlab)
    monkeypatch.setattr(orchestrator, "_installed_memory_gib", lambda: 64.0)
    bvals = np.loadtxt(subject_config.bvals).reshape(-1)
    metrics = StripeMetrics(
        a_si=np.asarray(csi, dtype=float),
        c_si=np.asarray(csi, dtype=float),
        shells=np.asarray(bvals, dtype=int),
        peak_sagittal=np.zeros(bvals.size, dtype=int),
    )
    monkeypatch.setattr(
        stripe_qc,
        "compute_stripe_indices",
        lambda image, values: metrics,
    )

    def fake_denoise(context) -> None:
        action_calls.append("denoise_leaf")
        pa_image = nib.load(context.config.dwi_pa)
        ap_image = nib.load(context.config.b0_ap)
        pa = np.asarray(pa_image.dataobj, dtype=np.float32)
        ap = np.asarray(ap_image.dataobj, dtype=np.float32)
        if ap.ndim == 3:
            ap = ap[..., None]
        raw_mean = np.mean(pa[..., bvals < 50.0], axis=-1)
        spatial = pa.shape[:3]
        for name, values, image in (
            ("denoised_PA.nii.gz", pa, pa_image),
            ("denoised_AP.nii.gz", ap, ap_image),
            ("sigma_PA.nii.gz", np.zeros(spatial), pa_image),
            ("sigma_AP.nii.gz", np.zeros(spatial), ap_image),
            ("raw_mean_b0.nii.gz", raw_mean, pa_image),
            ("raw_mean_b0_bet_mask.nii.gz", np.ones(spatial), pa_image),
            (
                "raw_mean_b0_bet_mask_dilated.nii.gz",
                np.ones(spatial),
                pa_image,
            ),
        ):
            _save_image(context.denoise_dir / name, values, image.affine)
        _write_all(context.denoise_dir / "denoise_metrics.json", "{}\n")
        _write_all(context.denoise_dir / "denoise_fsl.log")

    def fake_gibbs(context) -> None:
        action_calls.append("gibbs_leaf")
        for source_name, destination_name in (
            ("denoised_PA.nii.gz", "gibbs_PA.nii.gz"),
            ("denoised_AP.nii.gz", "gibbs_AP.nii.gz"),
        ):
            image = nib.load(context.denoise_dir / source_name)
            _save_image(
                context.gibbs_dir / destination_name,
                np.asarray(image.dataobj, dtype=np.float32),
                image.affine,
            )
        _write_all(context.gibbs_dir / "gibbs_metrics.json", "{}\n")

    monkeypatch.setattr(orchestrator, "run_denoise", fake_denoise)
    monkeypatch.setattr(orchestrator, "run_gibbs", fake_gibbs)

    def populate_jhu(work: Path) -> None:
        reference = nib.load(
            subject_config.subject_output / "06_dti" / "FA.nii.gz"
        )
        spatial = tuple(int(size) for size in reference.shape[:3])
        labels = np.zeros(spatial, dtype=np.int16)
        labels.reshape(-1)[:48] = np.arange(1, 49, dtype=np.int16)
        for name in (
            "FA_in_standard_affine.nii.gz",
            "FA_in_standard_nonlinear.nii.gz",
            "dti2standard_warp.nii.gz",
            "standard2dti_warp.nii.gz",
        ):
            _save_image(work / name, np.zeros(spatial), reference.affine)
        _save_image(work / "WM_JHU_ROIs.nii.gz", labels, reference.affine)
        np.savetxt(work / "dti2standard_affine.mat", np.eye(4))
        _write_all(work / "jhu_registration_fsl.log")

    def fake_fsl_command(argv, log_path: Path, environment):
        executable = Path(argv[0]).name
        action_calls.append(f"fsl:{executable}")
        assert Path(argv[0]).is_absolute()
        assert dict(environment) == runtime.fsl.environment
        work = Path(log_path).parent
        _write_all(Path(log_path), executable + "\n")
        audit = audit_inputs(subject_config)
        affine = np.asarray(audit.pa_affine)
        spatial = tuple(audit.pa_shape[:3])
        if executable == "topup":
            merged = nib.load(work / "PA_AP_b0.nii.gz")
            values = np.asarray(merged.dataobj, dtype=np.float32)
            _save_image(
                work / "topup_PA_AP_b0_fieldcoef.nii.gz",
                np.zeros((*spatial, 2)),
                affine,
            )
            _save_image(
                work / "topup_corrected_b0s.nii.gz",
                values + 0.5,
                affine,
            )
            _save_image(
                work / "topup_field_Hz.nii.gz",
                np.zeros(spatial),
                affine,
            )
            np.savetxt(
                work / "topup_PA_AP_b0_movpar.txt",
                np.zeros((values.shape[3], 6)),
            )
        elif executable == "applytopup":
            pytest.fail(
                "03_topup must not combine unequal PA/AP series with applytopup"
            )
        elif executable == "fslmaths":
            corrected = nib.load(Path(str(argv[1]) + ".nii.gz"))
            mean = np.mean(
                np.asarray(corrected.dataobj, dtype=np.float32),
                axis=-1,
            )
            _save_image(Path(str(argv[-1]) + ".nii.gz"), mean, affine)
        elif executable == "bet":
            source = nib.load(Path(str(argv[1]) + ".nii.gz"))
            values = np.asarray(source.dataobj, dtype=np.float32)
            prefix = Path(argv[2])
            _save_image(Path(str(prefix) + ".nii.gz"), values, source.affine)
            _save_image(
                Path(str(prefix) + "_mask.nii.gz"),
                np.ones(spatial, dtype=np.uint8),
                source.affine,
            )
        elif executable == "eddy":
            _populate_valid_eddy_core(work, subject_config)
        elif executable == "eddy_quad":
            quad = work / "eddy_quad"
            _populate_valid_quad(quad, work, subject_config)
        elif executable == "flirt":
            populate_jhu(work)
        return type(
            "SyntheticCommandResult",
            (),
            {"returncode": 0, "stderr": "", "stdout": ""},
        )()

    monkeypatch.setattr(orchestrator, "run_fsl_command", fake_fsl_command)

    reference = nib.load(subject_config.dwi_pa)
    spatial = tuple(int(size) for size in reference.shape[:3])

    def populate_model(context, names: tuple[str, ...], metrics_name: str) -> None:
        action_calls.append(metrics_name)
        for name in names:
            shape = (*spatial, 3) if name == "V1.nii.gz" else spatial
            _save_image(
                context.work_dir / name,
                np.ones(shape, dtype=np.float32),
                reference.affine,
            )
        _write_all(context.work_dir / metrics_name, "{}\n")

    monkeypatch.setattr(
        orchestrator,
        "fit_dti",
        lambda context: populate_model(
            context,
            ("FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz", "V1.nii.gz"),
            "dti_metrics.json",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "fit_dki",
        lambda context: populate_model(
            context,
            (
                "FA.nii.gz",
                "MD.nii.gz",
                "AD.nii.gz",
                "RD.nii.gz",
                "V1.nii.gz",
                "MK.nii.gz",
                "AK.nii.gz",
                "RK.nii.gz",
            ),
            "dki_metrics.json",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "fit_direct_dki",
        lambda context: populate_model(
            context,
            ("MD.nii.gz", "MK.nii.gz", "S0.nii.gz"),
            "dki_direct_metrics.json",
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "prepare_noddi",
        lambda context: action_calls.append("noddi_prepare_leaf"),
    )
    monkeypatch.setattr(
        orchestrator,
        "launch_noddi_workers",
        lambda context: action_calls.append("noddi_launch_leaf"),
    )

    def fake_merge_noddi(context) -> None:
        action_calls.append("noddi_merge_leaf")
        image_names = (
            "NODDI_odi.nii",
            "NODDI_ficvf.nii",
            "NODDI_fiso.nii",
            "NODDI_kappa.nii",
            "NODDI_fmin.nii",
            "NODDI_error_code.nii",
            "NODDI_fibredirs_xvec.nii",
            "NODDI_fibredirs_yvec.nii",
            "NODDI_fibredirs_zvec.nii",
        )
        for name in image_names:
            _save_image(
                context.stage_dir / name,
                np.zeros(spatial, dtype=np.float32),
                reference.affine,
            )
        for name, content in (
            ("NODDI_params.mat", "synthetic MAT\n"),
            ("noddi_metrics.json", "{}\n"),
            ("noddi_prepare.json", "{}\n"),
            ("noddi_prepare.log", "synthetic prepare\n"),
            ("merge_noddi.log", "synthetic merge\n"),
        ):
            _write_all(context.stage_dir / name, content)

    monkeypatch.setattr(orchestrator, "merge_noddi", fake_merge_noddi)

    def fake_qc(context) -> None:
        action_calls.append("qc_leaf")
        for filename in orchestrator.FIGURE_FILENAMES.values():
            _write_all(
                context.output_directory
                / filename.replace(
                    "${subject_id}", subject_config.subject_id
                ),
                "synthetic figure\n",
            )
        _write_all(context.output_directory / "qc_manifest.json", "{}\n")

    monkeypatch.setattr(orchestrator, "generate_all_qc", fake_qc)

    def fake_report(context) -> None:
        action_calls.append("report_leaf")
        detail_sizes = {
            path.name: len(path.read_bytes())
            for path in context.stripe_detail_files
        }
        _write_all(
            context.output_directory
            / f"{subject_config.subject_id}_QC_report.pdf",
            "synthetic PDF\n",
        )
        _write_all(
            context.output_directory
            / f"{subject_config.subject_id}_analysis_report.md",
            "synthetic report\n",
        )
        _write_all(
            context.output_directory
            / f"{subject_config.subject_id}_run_summary.json",
            json.dumps({"stripe_detail_sizes": detail_sizes}) + "\n",
        )

    monkeypatch.setattr(orchestrator, "write_final_report", fake_report)
    return runtime, action_calls, discoveries


def test_real_public_plan_include_with_flags_completes_then_skips_current_rerun(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, action_calls, discoveries = _install_real_plan_test_boundaries(
        monkeypatch,
        subject_config,
        tmp_path,
        np.asarray([1.26, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    first = run_pipeline(subject_config, "run")
    calls_after_first = tuple(action_calls)
    second = run_pipeline(subject_config, "run")
    audit = audit_inputs(subject_config)
    corrected = nib.load(
        subject_config.subject_output
        / "03_topup"
        / "topup_corrected_b0s.nii.gz"
    )
    decision = json.loads(
        (
            subject_config.subject_output
            / "00_pre_denoise_motion_qc"
            / "stripe_decision.json"
        ).read_text(encoding="utf-8")
    )

    assert first.status == "COMPLETE"
    assert len(audit.b0_indices) == 1
    assert audit.ap_b0_count == 2
    assert action_calls.count("fsl:topup") == 1
    assert "fsl:applytopup" not in action_calls
    assert corrected.shape == (*audit.pa_shape[:3], 3)
    assert not (
        subject_config.subject_output
        / "05_eddy"
        / "eddy_quad"
        / "vols_no_outliers.txt"
    ).exists()
    assert first.stage_statuses == tuple(
        (name, "completed") for name in STAGE_ORDER
    )
    assert len(first.stages) == 15
    assert all(
        (
            subject_config.subject_output
            / name
            / ".stage_complete.json"
        ).is_file()
        for name in STAGE_ORDER
    )
    assert second.status == "COMPLETE"
    assert second.stage_statuses == tuple(
        (name, "skipped") for name in STAGE_ORDER
    )
    assert tuple(action_calls) == calls_after_first
    assert discoveries == ["fsl", "matlab", "fsl", "matlab"]
    assert decision["decision"] == "INCLUDE_WITH_FLAGS"


def test_actual_denoise_action_uses_discovered_absolute_bet_and_isolated_env(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random = np.random.default_rng(20260727)
    for path in (subject_config.dwi_pa, subject_config.b0_ap):
        image = nib.load(path)
        values = 100.0 + 10.0 * random.standard_normal(image.shape)
        nib.save(
            nib.Nifti1Image(
                np.asarray(values, dtype=np.float32),
                image.affine,
                image.header,
            ),
            path,
        )
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    metrics = StripeMetrics(
        a_si=np.ones(8),
        c_si=np.ones(8),
        shells=np.loadtxt(subject_config.bvals).reshape(-1),
        peak_sagittal=np.zeros(8, dtype=int),
    )
    monkeypatch.setattr(
        stripe_qc,
        "compute_stripe_indices",
        lambda image, values: metrics,
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_fsl",
        lambda config: runtime.fsl,
    )
    captured: list[
        tuple[tuple[str, ...], Path, dict[str, str]]
    ] = []
    process_fsldir = os.environ.get("FSLDIR")

    def fake_early_bet(argv, log_path: Path, environment):
        captured.append((tuple(argv), Path(log_path), dict(environment)))
        source = nib.load(argv[1])
        values = np.asarray(source.dataobj, dtype=np.float32)
        prefix = Path(argv[2])
        _save_image(Path(str(prefix) + ".nii.gz"), values, source.affine)
        _save_image(
            Path(str(prefix) + "_mask.nii.gz"),
            np.ones(values.shape, dtype=np.uint8),
            source.affine,
        )
        _write_all(Path(log_path), "synthetic early BET\n")
        environment["SYNTHETIC_MUTATION"] = "must-not-escape"
        return type(
            "SyntheticCommandResult",
            (),
            {"returncode": 0, "stderr": "", "stdout": ""},
        )()

    monkeypatch.setattr(orchestrator, "run_fsl_command", fake_early_bet)
    denoise = {
        spec.name: spec for spec in build_plan(subject_config)
    }["01_denoise"]
    work = tmp_path / "actual-denoise"
    work.mkdir()

    denoise.action(work)
    outputs = tuple(denoise.validator(work))

    assert len(captured) == 1
    argv, log_path, environment = captured[0]
    assert argv == (
        str(runtime.fsl.bet),
        str(work / "raw_mean_b0.nii.gz"),
        str(work / "raw_mean_b0_bet"),
        "-R",
        "-f",
        "0.25",
        "-g",
        "0",
        "-m",
    )
    assert Path(argv[0]).is_absolute()
    assert log_path == work / "denoise_fsl.log"
    assert environment == runtime.fsl.environment
    assert "SYNTHETIC_MUTATION" not in runtime.fsl.environment
    assert os.environ.get("FSLDIR") == process_fsldir
    assert len(outputs) == 9
    assert all(path.is_file() for path in outputs)


def test_real_public_plan_hold_stops_before_runtime_discovery(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, action_calls, discoveries = _install_real_plan_test_boundaries(
        monkeypatch,
        subject_config,
        tmp_path,
        np.asarray([1.20, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_fsl",
        lambda config: pytest.fail("FSL discovery ran after a real QC hold"),
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_matlab",
        lambda config: pytest.fail("MATLAB discovery ran after a real QC hold"),
    )

    outcome = run_pipeline(subject_config, "run")
    decision = json.loads(
        (
            subject_config.subject_output
            / "00_pre_denoise_motion_qc"
            / "stripe_decision.json"
        ).read_text(encoding="utf-8")
    )

    assert outcome.status == "HOLD_FOR_REVIEW"
    assert len(outcome.stages) == 2
    assert decision["decision"] == "HOLD_FOR_REVIEW"
    assert discoveries == []
    assert action_calls == []
    assert len(
        list(
            subject_config.subject_output.glob(
                "*/.stage_complete.json"
            )
        )
    ) == 2


def _simple_plan(
    subject_config,
    decision: str,
    calls: list[str],
) -> list[StageSpec]:
    source = Path(orchestrator.__file__)
    root = subject_config.subject_output
    plan: list[StageSpec] = []
    previous: Path | None = None
    for name in STAGE_ORDER:
        output = root / name / "payload.txt"

        def action(work: Path, *, stage: str = name) -> None:
            calls.append(stage)
            (work / "payload.txt").write_text(stage + "\n", encoding="utf-8")
            if stage == "00_pre_denoise_motion_qc":
                exit_code = {
                    "EXCLUDE": 20,
                    "HOLD_FOR_REVIEW": 21,
                }.get(decision, 0)
                (work / "stripe_decision.json").write_text(
                    json.dumps(
                        {
                            "subject_id": subject_config.subject_id,
                            "decision": decision,
                            "exit_code": exit_code,
                            "ambiguous_reviewed": False,
                            "flagged_indices_zero_based": {
                                "high": [],
                                "ambiguous": [],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

        def validator(work: Path, *, stage: str = name) -> tuple[Path, ...]:
            outputs = [work / "payload.txt"]
            if stage == "00_pre_denoise_motion_qc":
                outputs.append(work / "stripe_decision.json")
            return tuple(outputs)

        inputs = (
            (subject_config.dwi_pa,)
            if previous is None
            else (previous,)
        )
        plan.append(StageSpec(name, action, validator, inputs, (source,)))
        previous = output
    return plan


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    subject_config,
    decision: str,
    calls: list[str],
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_build_plan",
        lambda config, runtime: _simple_plan(config, decision, calls),
    )
    monkeypatch.setattr(
        orchestrator._Runtime, "require_fsl", lambda self: object()
    )
    monkeypatch.setattr(
        orchestrator._Runtime, "require_matlab", lambda self: object()
    )
    monkeypatch.setattr(
        orchestrator, "_software_provenance", lambda runtime: {"full": "1"}
    )


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_count"),
    [
        ("HOLD_FOR_REVIEW", "HOLD_FOR_REVIEW", 2),
        ("INCLUDE_WITH_FLAGS", "COMPLETE", len(STAGE_ORDER)),
    ],
)
def test_fake_normal_run_honors_gate_and_completes_all_stages(
    subject_config,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    expected_status: str,
    expected_count: int,
) -> None:
    calls: list[str] = []
    _install_fake_pipeline(monkeypatch, subject_config, decision, calls)

    outcome = run_pipeline(subject_config, "run")

    assert outcome.status == expected_status
    assert calls == list(STAGE_ORDER[:expected_count])


def test_fake_normal_run_exact_current_skip_stale_partial_and_force_contracts(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _install_fake_pipeline(monkeypatch, subject_config, "INCLUDE", calls)
    first = run_pipeline(subject_config, "run")
    assert first.status == "COMPLETE"
    calls.clear()

    second = run_pipeline(subject_config, "run")
    assert calls == []
    assert all(status == "skipped" for _, status in second.stage_statuses)

    chosen = STAGE_ORDER.index("05_eddy")
    forced = run_pipeline(subject_config, "run", force_stage="05_eddy")
    assert all(
        status == "skipped" for _, status in forced.stage_statuses[:chosen]
    )
    assert all(
        status == "completed" for _, status in forced.stage_statuses[chosen:]
    )
    assert calls == list(STAGE_ORDER[chosen:])

    (subject_config.subject_output / "07_dki" / "payload.txt").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(StageStateError, match="stale|noncurrent"):
        run_pipeline(subject_config, "run")


def test_fake_run_refuses_non_noddi_partial_work(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _install_fake_pipeline(monkeypatch, subject_config, "INCLUDE", calls)
    partial = subject_config.subject_output / ".work" / "02_gibbs"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("partial\n", encoding="utf-8")

    with pytest.raises(StageStateError, match="partial"):
        run_pipeline(subject_config, "run")

    assert calls == list(STAGE_ORDER[:2])


def test_dry_run_uses_phase_provenance_and_never_resets_blocked_chain(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    calls: list[str] = []
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    plan = _simple_plan(subject_config, "INCLUDE", calls)
    root = subject_config.subject_output
    root.mkdir(parents=True)
    base_runner = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            orchestrator._base_software_provenance(),
        )
    )
    full_runner = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            _software_provenance(runtime),
        )
    )
    for spec in plan[:2]:
        base_runner.run(spec)
    full_runner.run(plan[2])
    monkeypatch.setattr(orchestrator, "_dry_run_commands", lambda *args: ())

    outcomes = _dry_run(
        subject_config, audit_inputs(subject_config), plan, runtime
    )
    assert [outcome.status for outcome in outcomes[:4]] == [
        "current/skipped",
        "current/skipped",
        "current/skipped",
        "runnable",
    ]

    stale = root / plan[3].name
    stale.mkdir()
    (stale / "unrecorded").write_text("stale\n", encoding="utf-8")
    noddi_work = root / ".work" / "08_noddi"
    noddi_work.mkdir(parents=True)
    (noddi_work / "checkpoint").write_text("partial\n", encoding="utf-8")
    outcomes = _dry_run(
        subject_config, audit_inputs(subject_config), plan, runtime
    )
    by_name = {outcome.stage: outcome.status for outcome in outcomes}
    assert by_name[plan[3].name] == "stale"
    assert by_name["08_noddi"] == "blocked"
    assert by_name["09_jhu_48roi"] == "blocked"
    capsys.readouterr()


@pytest.mark.parametrize("decision", ["HOLD_FOR_REVIEW", "EXCLUDE"])
def test_dry_run_propagates_current_stop_gate_to_all_scientific_stages(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    calls: list[str] = []
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    plan = _simple_plan(subject_config, decision, calls)
    root = subject_config.subject_output
    root.mkdir(parents=True)
    gate_runner = StageRunner(
        StageContext(
            subject_config,
            Path(orchestrator.__file__).parents[2],
            root,
            orchestrator._base_software_provenance(),
        )
    )
    for spec in plan[:2]:
        gate_runner.run(spec)
    monkeypatch.setattr(orchestrator, "_dry_run_commands", lambda *args: ())

    outcomes = _dry_run(
        subject_config, audit_inputs(subject_config), plan, runtime
    )

    assert [outcome.status for outcome in outcomes[:2]] == [
        "current/skipped",
        "current/skipped",
    ]
    assert all(outcome.status == "blocked" for outcome in outcomes[2:])


def test_full_dry_run_is_nonmutating_and_reports_sequentially_runnable_plan(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    monkeypatch.setattr(orchestrator, "_discover_runtime", lambda config: runtime)
    monkeypatch.setattr(
        orchestrator,
        "_build_plan",
        lambda config, selected_runtime: _simple_plan(
            config, "INCLUDE", calls
        ),
    )
    monkeypatch.setattr(orchestrator, "_dry_run_commands", lambda *args: ())

    outcome = run_pipeline(subject_config, "dry-run")

    assert outcome.status == "DRY_RUN"
    assert all(status == "runnable" for _, status in outcome.stage_statuses)
    assert calls == []
    assert not subject_config.subject_output.exists()


def test_dry_run_reports_dangling_final_symlink_as_stale_and_blocks_dependents(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = _fake_runtime(tmp_path, subject_config)
    plan = _simple_plan(subject_config, "INCLUDE", [])
    root = subject_config.subject_output
    root.mkdir(parents=True)
    (root / "02_gibbs").symlink_to(tmp_path / "missing-target")
    monkeypatch.setattr(orchestrator, "_dry_run_commands", lambda *args: ())

    outcomes = _dry_run(
        subject_config, audit_inputs(subject_config), plan, runtime
    )
    by_name = {outcome.stage: outcome.status for outcome in outcomes}

    assert by_name["02_gibbs"] == "stale"
    assert all(
        by_name[name] == "blocked"
        for name in STAGE_ORDER[STAGE_ORDER.index("03_topup") :]
    )


def test_subject_lock_rejects_parent_swap_without_locking_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_lock_anchor_root",
        lambda: tmp_path / "lock-anchors",
        raising=False,
    )
    parent = tmp_path / "parent"
    subject = parent / "subject"
    outside = tmp_path / "outside"
    subject.mkdir(parents=True)
    outside.mkdir()
    old_parent = tmp_path / "old-parent"
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "subject" and dir_fd is not None and not swapped:
            swapped = True
            parent.rename(old_parent)
            parent.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(orchestrator.os, "open", racing_open)

    with pytest.raises(StageStateError, match="identity|changed|symbolic"):
        with _SubjectLock(subject):
            pass

    assert swapped
    assert not (outside / ".pipeline.lock").exists()


def test_subject_lock_guard_prevents_different_inode_after_postcheck_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_lock_anchor_root",
        lambda: tmp_path / "lock-anchors",
        raising=False,
    )
    parent = tmp_path / "parent"
    subject = parent / "subject"
    subject.mkdir(parents=True)
    displaced = tmp_path / "displaced-parent"
    real_match = orchestrator._directory_path_matches
    calls = 0

    def swap_after_final_check(path: Path, expected: os.stat_result) -> bool:
        nonlocal calls
        result = real_match(path, expected)
        if path == subject:
            calls += 1
            if calls == 3 and result:
                parent.rename(displaced)
                subject.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        orchestrator, "_directory_path_matches", swap_after_final_check
    )

    with _SubjectLock(subject):
        with pytest.raises(StageStateError, match="already running"):
            with _SubjectLock(subject):
                pass

    assert (displaced / "subject" / ".pipeline.lock").is_file()
    assert not (subject / ".pipeline.lock").exists()
