from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from dmri_pipeline.summary import (
    CANONICAL_METRICS,
    COUNT_FIELDS,
    SummaryContext,
    SummaryError,
    build_summary_mask,
    summarize_subject,
)


EXPECTED_METRICS = (
    "DTI_FA",
    "DTI_MD",
    "DTI_AD",
    "DTI_RD",
    "DKI_FA",
    "DKI_MD",
    "DKI_AD",
    "DKI_RD",
    "DKI_MK",
    "DKI_AK",
    "DKI_RK",
    "DKI_DIRECT_MD",
    "DKI_DIRECT_MK",
    "DKI_DIRECT_S0",
    "NODDI_ODI",
    "NODDI_FICVF",
    "NODDI_FISO",
)
EXPECTED_COUNTS = (
    "atlas_labeled_voxel_count",
    "after_brain_mask_voxel_count",
    "after_noddi_success_voxel_count",
    "after_fiso_finite_voxel_count",
    "after_fiso_threshold_voxel_count",
    "common_mask_voxel_count",
)


def _save(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def summary_case(subject_config) -> dict[str, object]:
    subject_output = subject_config.subject_output
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shape = (2, 5, 6)
    atlas = np.ones(shape, dtype=np.int16)
    atlas.flat[:48] = np.arange(1, 49)
    atlas.flat[48] = 0
    brain = np.ones(shape, dtype=np.uint8)
    brain.flat[49] = 0
    errors = np.zeros(shape, dtype=np.int16)
    errors.flat[50] = 1
    errors.flat[55] = -1

    warped_atlas = _save(
        subject_output / "09_jhu_48roi" / "JHU_labels_subject.nii.gz",
        atlas,
        affine,
    )
    brain_mask = _save(
        subject_output / "04_bet" / "nodif_brain_mask_clean.nii.gz",
        brain,
        affine,
    )
    error_codes = _save(
        subject_output / "08_noddi" / "NODDI_error_code.nii.gz",
        errors,
        affine,
    )

    metric_maps: dict[str, Path] = {}
    for metric in EXPECTED_METRICS:
        data = np.full(shape, 10.0, dtype=np.float64)
        data.flat[[0, 54, 58, 59]] = [1.0, 3.0, 100.0, 5.0]
        if metric == "NODDI_FISO":
            data.fill(0.5)
            data.flat[51] = np.nan
            data.flat[52] = 0.9001
            data.flat[54] = 0.9
            data.flat[56] = np.inf
            data.flat[58] = 0.899
        if metric == "DTI_FA":
            data.flat[53] = np.nan
        if metric == "DKI_MK":
            data.flat[57] = np.inf
        group = metric.split("_", 1)[0].lower()
        metric_maps[metric] = _save(
            subject_output / f"maps_{group}" / f"{metric}.nii.gz", data, affine
        )

    resource_root = Path(__file__).parents[1] / "resources" / "jhu_48roi"
    return {
        "config": subject_config,
        "warped_atlas": warped_atlas,
        "brain_mask": brain_mask,
        "metric_maps": metric_maps,
        "noddi_error_codes": error_codes,
        "atlas_xml": resource_root / "JHU-labels.xml",
        "atlas_provenance": resource_root / "provenance.json",
        "output_directory": subject_output / "10_summary",
    }


@pytest.fixture
def summary_context(summary_case: dict[str, object]) -> SummaryContext:
    return SummaryContext(**summary_case)


def test_canonical_contract_has_exact_stable_keys_and_counts() -> None:
    assert CANONICAL_METRICS == EXPECTED_METRICS
    assert COUNT_FIELDS == EXPECTED_COUNTS


def test_summary_interfaces_are_publicly_exported() -> None:
    import dmri_pipeline

    assert dmri_pipeline.ResourceValidationError.__name__ == "ResourceValidationError"
    assert dmri_pipeline.AtlasValidation.__name__ == "AtlasValidation"
    assert dmri_pipeline.validate_jhu_resource.__name__ == "validate_jhu_resource"
    assert dmri_pipeline.SummaryError.__name__ == "SummaryError"
    assert dmri_pipeline.SummaryContext.__name__ == "SummaryContext"
    assert dmri_pipeline.SummaryOutputs.__name__ == "SummaryOutputs"
    assert dmri_pipeline.build_summary_mask.__name__ == "build_summary_mask"
    assert dmri_pipeline.summarize_subject.__name__ == "summarize_subject"


def test_summary_has_48_rows_and_one_global_row(
    summary_context: SummaryContext,
) -> None:
    outputs = summarize_subject(summary_context)
    roi = pd.read_csv(outputs.roi_csv)
    global_values = pd.read_csv(outputs.global_csv)
    assert roi["label_id"].tolist() == list(range(1, 49))
    assert len(global_values) == 1
    assert not roi.isna().any().any()


def test_common_mask_applies_every_filter_in_order_and_is_read_only(
    summary_context: SummaryContext,
) -> None:
    mask = build_summary_mask(summary_context)
    assert mask.dtype == np.bool_
    assert not mask.flags.writeable
    assert int(mask.sum()) == 51
    assert not mask.flat[48]  # atlas background
    assert not mask.flat[49]  # outside cleaned brain
    assert not mask.flat[50]  # NODDI error
    assert not mask.flat[51]  # nonfinite FISO
    assert not mask.flat[52]  # FISO > 0.9
    assert not mask.flat[53]  # nonfinite canonical metric
    assert mask.flat[54]  # inclusive FISO == 0.9
    assert not mask.flat[55]  # only error code zero succeeds
    assert not mask.flat[56]  # infinite FISO
    assert not mask.flat[57]  # another nonfinite canonical metric
    with pytest.raises(ValueError):
        mask.flat[0] = False


def test_outputs_use_common_mask_true_names_means_and_kurtosis_medians(
    summary_context: SummaryContext,
) -> None:
    outputs = summarize_subject(summary_context)
    roi = pd.read_csv(outputs.roi_csv)
    global_values = pd.read_csv(outputs.global_csv)
    label_one = roi.loc[roi["label_id"] == 1].iloc[0]
    assert label_one["voxel_count"] == 4
    assert label_one["DTI_FA"] == pytest.approx(27.25)
    assert label_one["DKI_MK"] == pytest.approx(4.0)
    assert label_one["DKI_AK"] == pytest.approx(4.0)
    assert label_one["DKI_RK"] == pytest.approx(4.0)
    assert label_one["DKI_DIRECT_MK"] == pytest.approx(4.0)
    assert label_one["DKI_DIRECT_S0"] == pytest.approx(27.25)
    assert label_one["NODDI_FISO"] == pytest.approx(0.69975)
    assert roi.loc[roi["label_id"] == 31, "label_name"].item().startswith(
        "Sagittal stratum"
    )
    assert roi.loc[roi["label_id"] == 45, "label_name"].item() == (
        "Uncinate fasciculus R"
    )
    assert roi.loc[roi["label_id"] == 46, "label_name"].item() == (
        "Uncinate fasciculus L"
    )
    assert global_values.loc[0, "DKI_MK"] == pytest.approx(10.0)
    assert global_values.loc[0, "DTI_FA"] == pytest.approx(579.0 / 51.0)


def test_outputs_have_exact_columns_counts_generic_names_and_relative_json_paths(
    summary_context: SummaryContext,
) -> None:
    outputs = summarize_subject(summary_context)
    assert outputs.roi_csv.name == "SYNTH001_JHU_48ROI_metrics.csv"
    assert outputs.global_csv.name == "SYNTH001_global_metrics.csv"
    assert outputs.summary_json.name == "SYNTH001_summary.json"
    roi = pd.read_csv(outputs.roi_csv)
    global_values = pd.read_csv(outputs.global_csv)
    assert tuple(roi.columns) == (
        "subject_id",
        "label_id",
        "label_name",
        "voxel_count",
        *EXPECTED_METRICS,
    )
    assert tuple(global_values.columns) == (
        "subject_id",
        *EXPECTED_COUNTS,
        *EXPECTED_METRICS,
    )
    assert global_values.loc[0, list(EXPECTED_COUNTS)].tolist() == [
        59,
        58,
        56,
        54,
        53,
        51,
    ]
    assert int(roi["voxel_count"].sum()) == 51

    raw_json = outputs.summary_json.read_text(encoding="utf-8")
    assert "NaN" not in raw_json
    assert "Infinity" not in raw_json
    assert str(Path.home()) not in raw_json
    assert ".." not in raw_json
    summary = json.loads(raw_json)
    assert list(summary["common_mask"]["counts"]) == list(EXPECTED_COUNTS)
    assert list(summary["aggregation_rules"]) == list(EXPECTED_METRICS)
    assert summary["aggregation_rules"]["DKI_DIRECT_MK"] == "median"
    assert summary["aggregation_rules"]["DKI_DIRECT_S0"] == "mean"
    assert [entry["index"] for entry in summary["atlas"]["label_mapping"]] == list(
        range(49)
    )
    assert summary["atlas"]["label_mapping"][45]["name"] == "Uncinate fasciculus R"
    assert summary["noddi_error_code_histogram"] == {
        "scope": "cleaned_brain_mask",
        "bins": [
            {"error_code": -1, "voxel_count": 1},
            {"error_code": 0, "voxel_count": 57},
            {"error_code": 1, "voxel_count": 1},
        ],
    }
    assert list(summary["metric_maps"]) == list(EXPECTED_METRICS)
    for relative_path in summary["metric_maps"].values():
        assert not Path(relative_path).is_absolute()
        assert ".." not in Path(relative_path).parts


@pytest.mark.parametrize("bad_labels", ["missing", "extra", "fractional"])
def test_subject_space_atlas_requires_exact_integer_labels_1_to_48(
    summary_case: dict[str, object], bad_labels: str
) -> None:
    path = summary_case["warped_atlas"]
    assert isinstance(path, Path)
    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=float)
    if bad_labels == "missing":
        data[data == 48] = 47
    elif bad_labels == "extra":
        data.flat[48] = 49
    else:
        data[data == 1] = 1.5
    nib.save(nib.Nifti1Image(data, image.affine), path)
    with pytest.raises(SummaryError, match="warped atlas"):
        SummaryContext(**summary_case)


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        ("brain_mask", "affine", "grid"),
        ("noddi_error_codes", "fractional", "integer"),
        ("noddi_error_codes", "nonfinite", "finite"),
        ("metric", "shape", "grid"),
        ("metric", "affine", "grid"),
    ],
)
def test_context_rejects_grid_affine_and_error_code_violations(
    summary_case: dict[str, object],
    target: str,
    mutation: str,
    message: str,
) -> None:
    if target == "metric":
        metric_maps = summary_case["metric_maps"]
        assert isinstance(metric_maps, dict)
        path = metric_maps["DTI_MD"]
    else:
        path = summary_case[target]
    assert isinstance(path, Path)
    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=float)
    affine = image.affine.copy()
    if mutation == "affine":
        affine[0, 3] += 0.01
    elif mutation == "fractional":
        data.flat[0] = 0.5
    elif mutation == "nonfinite":
        data.flat[0] = np.nan
    elif mutation == "shape":
        data = data[:-1, :, :]
    nib.save(nib.Nifti1Image(data, affine), path)
    with pytest.raises(SummaryError, match=message):
        SummaryContext(**summary_case)


def test_context_requires_exact_metric_keys_and_normalizes_canonical_order(
    summary_case: dict[str, object],
) -> None:
    metric_maps = summary_case["metric_maps"]
    assert isinstance(metric_maps, dict)
    missing = dict(metric_maps)
    missing.pop("DKI_DIRECT_S0")
    with pytest.raises(SummaryError, match="canonical metric keys"):
        SummaryContext(**{**summary_case, "metric_maps": missing})
    extra = {**metric_maps, "IFOF": next(iter(metric_maps.values()))}
    with pytest.raises(SummaryError, match="canonical metric keys"):
        SummaryContext(**{**summary_case, "metric_maps": extra})
    reversed_maps = dict(reversed(metric_maps.items()))
    context = SummaryContext(**{**summary_case, "metric_maps": reversed_maps})
    assert tuple(context.metric_maps) == EXPECTED_METRICS


def test_zero_common_mask_voxel_for_any_roi_fails(
    summary_case: dict[str, object],
) -> None:
    path = summary_case["noddi_error_codes"]
    atlas_path = summary_case["warped_atlas"]
    assert isinstance(path, Path)
    assert isinstance(atlas_path, Path)
    errors_image = nib.load(path)
    errors = np.asarray(errors_image.dataobj, dtype=float)
    atlas = np.asarray(nib.load(atlas_path).dataobj)
    errors[atlas == 48] = 7
    nib.save(nib.Nifti1Image(errors, errors_image.affine), path)
    context = SummaryContext(**summary_case)
    with pytest.raises(SummaryError, match="label 48.*zero"):
        summarize_subject(context)
    output_directory = summary_case["output_directory"]
    assert isinstance(output_directory, Path)
    assert not output_directory.exists()


def test_output_overwrite_and_symlink_are_refused(
    summary_context: SummaryContext, tmp_path: Path
) -> None:
    summary_context.output_directory.mkdir()
    existing = (
        summary_context.output_directory
        / f"{summary_context.config.subject_id}_global_metrics.csv"
    )
    existing.write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="already exists"):
        summarize_subject(summary_context)
    assert existing.read_text(encoding="utf-8") == "do not replace\n"
    existing.unlink()
    dangling = summary_context.output_directory / (
        f"{summary_context.config.subject_id}_summary.json"
    )
    dangling.symlink_to(tmp_path / "outside.json")
    with pytest.raises(SummaryError, match="already exists"):
        summarize_subject(summary_context)
    assert dangling.is_symlink()


def test_context_rejects_symlinked_output_and_input_aliases(
    summary_case: dict[str, object], tmp_path: Path
) -> None:
    output_directory = summary_case["output_directory"]
    assert isinstance(output_directory, Path)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(SummaryError, match="symlink"):
        SummaryContext(**summary_case)
    output_directory.unlink()

    metric_maps = summary_case["metric_maps"]
    assert isinstance(metric_maps, dict)
    alias = metric_maps["DTI_MD"]
    alias.unlink()
    os.link(metric_maps["DTI_FA"], alias)
    with pytest.raises(SummaryError, match="alias"):
        SummaryContext(**summary_case)


def test_summary_preserves_all_input_bytes(summary_context: SummaryContext) -> None:
    inputs = [
        summary_context.warped_atlas,
        summary_context.brain_mask,
        summary_context.noddi_error_codes,
        summary_context.atlas_xml,
        summary_context.atlas_provenance,
        *summary_context.metric_maps.values(),
    ]
    before = {path: _sha256(path) for path in inputs}
    summarize_subject(summary_context)
    assert {_path: _sha256(_path) for _path in inputs} == before


def test_summary_rejects_metric_mutation_at_output_boundary(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    original_write = summary_module._write_outputs_no_clobber

    def mutate_then_write(*args, **kwargs):
        path = summary_context.metric_maps["DTI_MD"]
        image = nib.load(path)
        changed = np.asarray(image.dataobj, dtype=float) + 1.0
        nib.save(nib.Nifti1Image(changed, image.affine), path)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        summary_module, "_write_outputs_no_clobber", mutate_then_write
    )
    with pytest.raises(SummaryError, match="changed during summary computation"):
        summarize_subject(summary_context)
    assert not summary_context.output_directory.exists()


def test_output_directory_swap_cannot_split_or_falsify_outputs(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    real_fsync = summary_module.os.fsync
    moved_directory = summary_context.output_directory.with_name("detached_summary")
    swapped = False

    def swap_after_first_file(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        if not swapped:
            swapped = True
            summary_context.output_directory.rename(moved_directory)
            summary_context.output_directory.mkdir()

    monkeypatch.setattr(summary_module.os, "fsync", swap_after_first_file)
    with pytest.raises(SummaryError, match="output directory changed"):
        summarize_subject(summary_context)
    assert list(moved_directory.iterdir()) == []
    assert list(summary_context.output_directory.iterdir()) == []


def test_rollback_keeps_replacement_inode_and_removes_own_partial_file(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    real_fsync = summary_module.os.fsync
    outputs = (
        summary_context.output_directory
        / f"{summary_context.config.subject_id}_JHU_48ROI_metrics.csv",
        summary_context.output_directory
        / f"{summary_context.config.subject_id}_global_metrics.csv",
    )
    calls = 0

    def replace_first_then_fail_second(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_fsync(descriptor)
            return
        outputs[0].unlink()
        outputs[0].write_text("replacement inode\n", encoding="utf-8")
        raise OSError("forced second fsync failure")

    monkeypatch.setattr(
        summary_module.os, "fsync", replace_first_then_fail_second
    )
    with pytest.raises(SummaryError, match="cannot write summary output"):
        summarize_subject(summary_context)
    assert outputs[0].read_text(encoding="utf-8") == "replacement inode\n"
    assert not outputs[1].exists()


def test_first_fsync_failure_rolls_back_and_retry_succeeds(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    real_fsync = summary_module.os.fsync
    failed = False

    def fail_once(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("forced first fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(summary_module.os, "fsync", fail_once)
    with pytest.raises(SummaryError, match="cannot write summary output"):
        summarize_subject(summary_context)
    assert list(summary_context.output_directory.iterdir()) == []
    outputs = summarize_subject(summary_context)
    assert outputs.roi_csv.is_file()
    assert outputs.global_csv.is_file()
    assert outputs.summary_json.is_file()


def test_first_write_failure_rolls_back_and_retry_succeeds(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    real_fdopen = summary_module.os.fdopen
    failed = False

    class FailFirstWrite:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, _content: str) -> int:
            raise OSError("forced first write failure")

        def flush(self) -> None:
            self.handle.flush()

        def fileno(self) -> int:
            return self.handle.fileno()

    def fail_write_once(*args, **kwargs):
        nonlocal failed
        handle = real_fdopen(*args, **kwargs)
        if not failed:
            failed = True
            return FailFirstWrite(handle)
        return handle

    monkeypatch.setattr(summary_module.os, "fdopen", fail_write_once)
    with pytest.raises(SummaryError, match="cannot write summary output"):
        summarize_subject(summary_context)
    assert list(summary_context.output_directory.iterdir()) == []
    outputs = summarize_subject(summary_context)
    assert outputs.roi_csv.is_file()
    assert outputs.global_csv.is_file()
    assert outputs.summary_json.is_file()


def test_temporary_metric_mutation_and_restore_cannot_change_loaded_values(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    target = summary_context.metric_maps["DTI_MD"]
    original_bytes = target.read_bytes()
    image = nib.load(target)
    replacement = target.with_name("temporary_replacement.nii.gz")
    nib.save(
        nib.Nifti1Image(
            np.full(image.shape, 999.0, dtype=np.float64), image.affine
        ),
        replacement,
    )
    replacement_bytes = replacement.read_bytes()
    replacement.unlink()
    real_load = summary_module._load_nifti
    injected = False

    def mutate_during_load(path: Path, label: str):
        nonlocal injected
        if label == "DTI_MD" and not injected:
            injected = True
            target.write_bytes(replacement_bytes)
            try:
                return real_load(path, label)
            finally:
                target.write_bytes(original_bytes)
        return real_load(path, label)

    monkeypatch.setattr(summary_module, "_load_nifti", mutate_during_load)
    outputs = summarize_subject(summary_context)
    global_values = pd.read_csv(outputs.global_csv)
    assert global_values.loc[0, "DTI_MD"] == pytest.approx(579.0 / 51.0)
    assert target.read_bytes() == original_bytes


def test_input_replacement_after_final_verifier_is_rejected_and_rolled_back(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    target = summary_context.metric_maps["DTI_AD"]
    replacement = target.with_name("same_bytes_new_inode.nii.gz")
    real_verify = summary_module._verify_input_hashes
    replaced = False

    def verify_then_replace(context, expected) -> None:
        nonlocal replaced
        real_verify(context, expected)
        if not replaced:
            replaced = True
            shutil.copyfile(target, replacement)
            os.replace(replacement, target)

    monkeypatch.setattr(
        summary_module, "_verify_input_hashes", verify_then_replace
    )
    with pytest.raises(SummaryError, match="changed during summary computation"):
        summarize_subject(summary_context)
    assert (
        not summary_context.output_directory.exists()
        or list(summary_context.output_directory.iterdir()) == []
    )


@pytest.mark.parametrize("replacement_kind", ["regular", "symlink", "hardlink", "fifo"])
def test_commit_rejects_foreign_output_replacement_and_retry_is_clean(
    summary_context: SummaryContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement_kind: str,
) -> None:
    import dmri_pipeline.summary as summary_module

    outputs = (
        summary_context.output_directory
        / f"{summary_context.config.subject_id}_JHU_48ROI_metrics.csv",
        summary_context.output_directory
        / f"{summary_context.config.subject_id}_global_metrics.csv",
        summary_context.output_directory
        / f"{summary_context.config.subject_id}_summary.json",
    )
    foreign = tmp_path / "foreign-output"
    foreign.write_text("foreign bytes\n", encoding="utf-8")
    real_fsync = summary_module.os.fsync
    injected = False

    def replace_after_first_fsync(descriptor: int) -> None:
        nonlocal injected
        real_fsync(descriptor)
        if injected:
            return
        injected = True
        outputs[0].unlink()
        if replacement_kind == "regular":
            outputs[0].write_text("foreign bytes\n", encoding="utf-8")
        elif replacement_kind == "symlink":
            outputs[0].symlink_to(foreign)
        elif replacement_kind == "hardlink":
            os.link(foreign, outputs[0])
        else:
            os.mkfifo(outputs[0])

    monkeypatch.setattr(summary_module.os, "fsync", replace_after_first_fsync)
    with pytest.raises(SummaryError, match="summary output changed during commit"):
        summarize_subject(summary_context)

    metadata = outputs[0].lstat()
    if replacement_kind == "regular":
        assert stat.S_ISREG(metadata.st_mode)
        assert outputs[0].read_text(encoding="utf-8") == "foreign bytes\n"
    elif replacement_kind == "symlink":
        assert outputs[0].is_symlink()
        assert outputs[0].readlink() == foreign
    elif replacement_kind == "hardlink":
        assert stat.S_ISREG(metadata.st_mode)
        assert metadata.st_ino == foreign.stat().st_ino
        assert metadata.st_nlink == 2
    else:
        assert stat.S_ISFIFO(metadata.st_mode)
    assert not outputs[1].exists()
    assert not outputs[2].exists()

    outputs[0].unlink()
    retried = summarize_subject(summary_context)
    assert retried.roi_csv.is_file()
    assert retried.global_csv.is_file()
    assert retried.summary_json.is_file()


def test_final_input_verify_failure_rolls_back_outputs_and_retry_is_clean(
    summary_context: SummaryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.summary as summary_module

    target = summary_context.metric_maps["DTI_RD"]
    replacement = target.with_name("same-bytes-final-verify.nii.gz")
    real_verify = summary_module._InputSnapshots.verify
    calls = 0

    def replace_on_final_verify(snapshots) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            shutil.copyfile(target, replacement)
            os.replace(replacement, target)
        real_verify(snapshots)

    monkeypatch.setattr(
        summary_module._InputSnapshots, "verify", replace_on_final_verify
    )
    with pytest.raises(SummaryError, match="changed during summary computation"):
        summarize_subject(summary_context)
    assert list(summary_context.output_directory.iterdir()) == []

    retried = summarize_subject(summary_context)
    assert retried.roi_csv.is_file()
    assert retried.global_csv.is_file()
    assert retried.summary_json.is_file()
