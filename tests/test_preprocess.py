from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dmri_pipeline.audit import audit_inputs
from dmri_pipeline.config import AcquisitionConfig
from dmri_pipeline.preprocess import (
    PreprocessContext,
    PreprocessError,
    choose_patch_radius,
    clean_bet_mask,
    prepare_topup_inputs,
    run_denoise,
    run_gibbs,
)
from dmri_pipeline.utils import sha256_file


def _save_image(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), path)


def _make_context(
    tmp_path: Path,
    subject_config,
    *,
    command_executor=None,
) -> PreprocessContext:
    kwargs = {}
    if command_executor is not None:
        kwargs["command_executor"] = command_executor
    return PreprocessContext(
        config=subject_config,
        audit=audit_inputs(subject_config),
        denoise_dir=tmp_path / "01_denoise.work",
        gibbs_dir=tmp_path / "02_gibbs.work",
        topup_dir=tmp_path / "03_topup.work",
        bet_dir=tmp_path / "04_bet.work",
        bet_mask_source=tmp_path / "04_bet.work" / "nodif_brain_mask_raw.nii.gz",
        **kwargs,
    )


@pytest.fixture
def fake_bet_executor():
    calls: list[tuple[str, ...]] = []

    def execute(argv):
        calls.append(tuple(argv))
        source = nib.load(argv[1])
        mask = np.zeros(source.shape, dtype=np.uint8)
        mask[4, 4, 4] = 1
        _save_image(Path(f"{argv[2]}_mask.nii.gz"), mask, source.affine)
        return SimpleNamespace(returncode=0, stderr="")

    execute.calls = calls
    return execute


@pytest.fixture
def preprocess_context(tmp_path, subject_config, fake_bet_executor):
    return _make_context(
        tmp_path,
        subject_config,
        command_executor=fake_bet_executor,
    )


@pytest.mark.parametrize(
    ("n", "radius"),
    [(1, 1), (7, 1), (26, 1), (27, 2), (124, 2), (125, 3), (177, 3)],
)
def test_choose_patch_radius_has_more_voxels_than_measurements(n, radius):
    assert choose_patch_radius(n) == radius
    assert (2 * radius + 1) ** 3 > n
    if radius:
        assert (2 * (radius - 1) + 1) ** 3 <= n


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "8", None])
def test_choose_patch_radius_rejects_nonpositive_bool_and_noninteger_counts(invalid):
    with pytest.raises(PreprocessError, match="positive integer"):
        choose_patch_radius(invalid)


def test_context_is_immutable_and_rejects_overlapping_stage_directories(
    preprocess_context,
):
    with pytest.raises((AttributeError, TypeError)):
        preprocess_context.topup_dir = preprocess_context.gibbs_dir
    with pytest.raises(PreprocessError, match="distinct"):
        replace(preprocess_context, topup_dir=preprocess_context.gibbs_dir)


def test_context_rejects_nonfinite_manually_constructed_acquisition(
    preprocess_context,
):
    invalid_acquisition = replace(
        preprocess_context.config.acquisition,
        total_readout_time=float("nan"),
    )
    invalid_config = replace(
        preprocess_context.config,
        acquisition=invalid_acquisition,
    )
    with pytest.raises(PreprocessError, match="readout"):
        replace(preprocess_context, config=invalid_config)


@pytest.mark.parametrize("invalid_readout", [True, False, "0.1", object()])
def test_context_normalizes_invalid_readout_types_to_preprocess_error(
    preprocess_context,
    invalid_readout,
):
    invalid_acquisition = replace(
        preprocess_context.config.acquisition,
        total_readout_time=invalid_readout,
    )
    invalid_config = replace(
        preprocess_context.config,
        acquisition=invalid_acquisition,
    )

    with pytest.raises(PreprocessError, match="readout.*real number"):
        replace(preprocess_context, config=invalid_config)


@pytest.mark.parametrize("invalid_axis", [True, False, "2", object()])
def test_context_normalizes_invalid_slice_axis_types_to_preprocess_error(
    preprocess_context,
    invalid_axis,
):
    invalid_acquisition = replace(
        preprocess_context.config.acquisition,
        slice_axis=invalid_axis,
    )
    invalid_config = replace(
        preprocess_context.config,
        acquisition=invalid_acquisition,
    )

    with pytest.raises(PreprocessError, match="slice axis.*integer"):
        replace(preprocess_context, config=invalid_config)


def test_context_rejects_symlinked_stage_work_directory(
    tmp_path,
    subject_config,
    fake_bet_executor,
):
    outside_directory = tmp_path / "outside-stage"
    outside_directory.mkdir()
    (tmp_path / "01_denoise.work").symlink_to(
        outside_directory,
        target_is_directory=True,
    )

    with pytest.raises(PreprocessError, match="symbolic link"):
        _make_context(
            tmp_path,
            subject_config,
            command_executor=fake_bet_executor,
        )


def test_context_rejects_symlinked_parent_component(
    preprocess_context,
    tmp_path,
):
    outside_parent = tmp_path / "outside-parent"
    outside_parent.mkdir()
    redirecting_parent = tmp_path / "redirecting-parent"
    redirecting_parent.symlink_to(outside_parent, target_is_directory=True)

    with pytest.raises(PreprocessError, match="symbolic link"):
        replace(
            preprocess_context,
            gibbs_dir=redirecting_parent / "02_gibbs.work",
        )


def test_context_rejects_symlinked_bet_mask_source(
    tmp_path,
    subject_config,
    fake_bet_executor,
):
    bet_directory = tmp_path / "04_bet.work"
    bet_directory.mkdir()
    outside_mask = tmp_path / "outside-mask.nii.gz"
    _save_image(outside_mask, np.ones((8, 8, 8), dtype=np.uint8), np.eye(4))
    (bet_directory / "nodif_brain_mask_raw.nii.gz").symlink_to(outside_mask)

    with pytest.raises(PreprocessError, match="symbolic link"):
        _make_context(
            tmp_path,
            subject_config,
            command_executor=fake_bet_executor,
        )


def test_context_rejects_symlink_then_parent_traversal_stage_alias_without_action(
    tmp_path,
    subject_config,
    fake_bet_executor,
    monkeypatch,
):
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    nested_outside = outside / "nested"
    inside.mkdir()
    nested_outside.mkdir(parents=True)
    (inside / "link").symlink_to(nested_outside, target_is_directory=True)
    traversal_stage = inside / "link" / ".." / "stage"
    direct_stage_alias = outside / "stage"
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (
            np.asarray(data, dtype=np.float32),
            np.ones(data.shape[:3], dtype=np.float32),
        ),
    )

    with pytest.raises(PreprocessError, match="parent traversal"):
        context = PreprocessContext(
            config=subject_config,
            audit=audit_inputs(subject_config),
            denoise_dir=traversal_stage,
            gibbs_dir=direct_stage_alias,
            topup_dir=tmp_path / "safe-topup",
            bet_dir=tmp_path / "safe-bet",
            bet_mask_source=tmp_path / "safe-bet" / "raw-mask.nii.gz",
            command_executor=fake_bet_executor,
        )
        run_denoise(context)

    assert fake_bet_executor.calls == []
    assert not direct_stage_alias.exists()


def test_context_rejects_symlink_then_parent_traversal_bet_source_without_reading(
    tmp_path,
    subject_config,
    fake_bet_executor,
):
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    nested_outside = outside / "nested"
    inside.mkdir()
    nested_outside.mkdir(parents=True)
    (inside / "link").symlink_to(nested_outside, target_is_directory=True)
    outside_mask = outside / "raw-mask.nii.gz"
    _save_image(
        outside_mask,
        np.ones((8, 8, 8), dtype=np.uint8),
        np.eye(4),
    )
    original_hash = sha256_file(outside_mask)
    traversal_source = inside / "link" / ".." / "raw-mask.nii.gz"
    safe_bet_dir = tmp_path / "safe-bet"

    with pytest.raises(PreprocessError, match="parent traversal"):
        context = PreprocessContext(
            config=subject_config,
            audit=audit_inputs(subject_config),
            denoise_dir=tmp_path / "safe-denoise",
            gibbs_dir=tmp_path / "safe-gibbs",
            topup_dir=tmp_path / "safe-topup",
            bet_dir=safe_bet_dir,
            bet_mask_source=traversal_source,
            command_executor=fake_bet_executor,
        )
        clean_bet_mask(context)

    assert sha256_file(outside_mask) == original_hash
    assert not safe_bet_dir.exists()


def test_context_preserves_safe_relative_nonexistent_paths(
    tmp_path,
    subject_config,
    fake_bet_executor,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    context = PreprocessContext(
        config=subject_config,
        audit=audit_inputs(subject_config),
        denoise_dir=Path("work/01_denoise.work"),
        gibbs_dir=Path("work/02_gibbs.work"),
        topup_dir=Path("work/03_topup.work"),
        bet_dir=Path("work/04_bet.work"),
        bet_mask_source=Path("work/04_bet.work/raw-mask.nii.gz"),
        command_executor=fake_bet_executor,
    )

    assert context.denoise_dir == Path("work/01_denoise.work")
    assert context.bet_mask_source == Path("work/04_bet.work/raw-mask.nii.gz")


def test_run_denoise_normalizes_3d_ap_and_uses_required_mppca_arguments(
    tmp_path,
    subject_config,
    fake_bet_executor,
    monkeypatch,
):
    ap = nib.load(subject_config.b0_ap)
    _save_image(subject_config.b0_ap, np.asanyarray(ap.dataobj)[..., 0], ap.affine)
    context = _make_context(
        tmp_path,
        subject_config,
        command_executor=fake_bet_executor,
    )
    calls = []

    def fake_mppca(data, **kwargs):
        calls.append((data.shape, kwargs.copy()))
        return np.asarray(data, dtype=np.float32) + 10.0, np.ones(data.shape[:3])

    monkeypatch.setattr("dmri_pipeline.preprocess.mppca", fake_mppca)
    details = run_denoise(context)

    assert [shape for shape, _ in calls] == [(8, 8, 8, 8), (8, 8, 8, 1)]
    assert calls[0][1]["patch_radius"] == 1
    assert calls[1][1]["patch_radius"] == 2
    for _, kwargs in calls:
        assert kwargs["pca_method"] == "eig"
        assert kwargs["return_sigma"] is True
        assert kwargs["out_dtype"] is np.float32
    assert calls[0][1]["mask"].shape == (8, 8, 8)
    assert calls[1][1]["mask"] is None
    assert nib.load(details.denoised_ap).shape == (8, 8, 8, 1)


def test_run_denoise_accepts_4d_ap_and_runs_bet_as_argv(
    preprocess_context,
    fake_bet_executor,
    monkeypatch,
):
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (
            np.asarray(data, dtype=np.float32),
            np.ones(data.shape[:3], dtype=np.float32),
        ),
    )
    details = run_denoise(preprocess_context)

    assert nib.load(details.denoised_ap).shape == (8, 8, 8, 2)
    assert len(fake_bet_executor.calls) == 1
    argv = fake_bet_executor.calls[0]
    assert isinstance(argv, tuple)
    assert argv[-6:] == ("-R", "-f", "0.25", "-g", "0", "-m")
    assert " " not in argv[0]
    assert Path(f"{argv[2]}_mask.nii.gz") == details.raw_bet_mask


def test_run_denoise_dilates_pa_mask_exactly_two_26_connected_iterations(
    preprocess_context,
    monkeypatch,
):
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (
            np.asarray(data, dtype=np.float32),
            np.ones(data.shape[:3], dtype=np.float32),
        ),
    )
    details = run_denoise(preprocess_context)

    mask = np.asanyarray(nib.load(details.dilated_mask).dataobj)
    expected = np.zeros((8, 8, 8), dtype=np.uint8)
    expected[2:7, 2:7, 2:7] = 1
    np.testing.assert_array_equal(mask, expected)


def test_run_denoise_restores_pa_outside_mask_and_does_not_mutate_raw_inputs(
    preprocess_context,
    monkeypatch,
):
    input_hashes = {
        path: sha256_file(path)
        for path in (
            preprocess_context.config.dwi_pa,
            preprocess_context.config.b0_ap,
            preprocess_context.config.bvals,
            preprocess_context.config.bvecs,
        )
    }

    def zeros_mppca(data, **kwargs):
        output = np.zeros_like(data, dtype=np.float32)
        data[...] = -5
        return output, np.ones(data.shape[:3])

    monkeypatch.setattr("dmri_pipeline.preprocess.mppca", zeros_mppca)
    details = run_denoise(preprocess_context)

    raw = np.asanyarray(nib.load(preprocess_context.config.dwi_pa).dataobj)
    denoised = np.asanyarray(nib.load(details.denoised_pa).dataobj)
    mask = np.asanyarray(nib.load(details.dilated_mask).dataobj).astype(bool)
    np.testing.assert_array_equal(denoised[~mask], raw[~mask])
    assert all(sha256_file(path) == digest for path, digest in input_hashes.items())


def test_run_denoise_preserves_geometry_dtype_descriptions_and_metrics(
    preprocess_context,
    monkeypatch,
):
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (
            np.asarray(data, dtype=np.float32),
            np.ones(data.shape[:3], dtype=np.float32),
        ),
    )
    details = run_denoise(preprocess_context)

    source = nib.load(preprocess_context.config.dwi_pa)
    output = nib.load(details.denoised_pa)
    np.testing.assert_array_equal(output.affine, source.affine)
    assert output.get_data_dtype() == np.dtype(np.float32)
    assert bytes(output.header["descrip"]).rstrip(b"\x00")
    raw_mask = nib.load(details.raw_bet_mask)
    assert raw_mask.get_data_dtype() == np.dtype(np.uint8)
    assert bytes(raw_mask.header["descrip"]).rstrip(b"\x00")
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics["pa_patch_radius"] == 1
    assert metrics["pa_measurement_count"] == 8
    assert metrics["ap_measurement_count"] == 2


def test_run_denoise_rejects_nonfinite_mppca_output(
    preprocess_context,
    monkeypatch,
):
    def nonfinite_mppca(data, **kwargs):
        output = np.asarray(data, dtype=np.float32).copy()
        output.flat[0] = np.nan
        return output, np.ones(data.shape[:3], dtype=np.float32)

    monkeypatch.setattr("dmri_pipeline.preprocess.mppca", nonfinite_mppca)
    with pytest.raises(PreprocessError, match="finite"):
        run_denoise(preprocess_context)


def test_run_denoise_refuses_preexisting_destination(
    preprocess_context,
    monkeypatch,
):
    preprocess_context.denoise_dir.mkdir()
    destination = preprocess_context.denoise_dir / "denoised_PA.nii.gz"
    destination.write_bytes(b"do not overwrite")
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (data, np.ones(data.shape[:3])),
    )
    with pytest.raises(PreprocessError, match="already exists"):
        run_denoise(preprocess_context)
    assert destination.read_bytes() == b"do not overwrite"


def test_run_denoise_refuses_dangling_destination_symlink_without_following_it(
    preprocess_context,
    monkeypatch,
    tmp_path,
):
    preprocess_context.denoise_dir.mkdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_target = outside_directory / "escaped-output.nii.gz"
    destination = preprocess_context.denoise_dir / "denoised_PA.nii.gz"
    destination.symlink_to(outside_target)
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (
            np.asarray(data, dtype=np.float32),
            np.ones(data.shape[:3], dtype=np.float32),
        ),
    )

    with pytest.raises(PreprocessError, match="already exists"):
        run_denoise(preprocess_context)

    assert os.path.lexists(destination)
    assert destination.is_symlink()
    assert not outside_target.exists()


def test_run_denoise_revalidates_parent_chain_before_writing(
    tmp_path,
    subject_config,
    fake_bet_executor,
    monkeypatch,
):
    work_parent = tmp_path / "work-parent"
    context = PreprocessContext(
        config=subject_config,
        audit=audit_inputs(subject_config),
        denoise_dir=work_parent / "01_denoise.work",
        gibbs_dir=work_parent / "02_gibbs.work",
        topup_dir=work_parent / "03_topup.work",
        bet_dir=work_parent / "04_bet.work",
        bet_mask_source=(
            work_parent / "04_bet.work" / "nodif_brain_mask_raw.nii.gz"
        ),
        command_executor=fake_bet_executor,
    )
    outside_parent = tmp_path / "outside-work-parent"
    outside_parent.mkdir()
    work_parent.symlink_to(outside_parent, target_is_directory=True)
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (
            np.asarray(data, dtype=np.float32),
            np.ones(data.shape[:3], dtype=np.float32),
        ),
    )

    with pytest.raises(PreprocessError, match="symbolic link"):
        run_denoise(context)

    assert list(outside_parent.iterdir()) == []


def test_run_denoise_surfaces_bet_failure_actionably(
    tmp_path,
    subject_config,
    monkeypatch,
):
    context = _make_context(
        tmp_path,
        subject_config,
        command_executor=lambda argv: SimpleNamespace(
            returncode=9, stderr="mask creation failed"
        ),
    )
    monkeypatch.setattr(
        "dmri_pipeline.preprocess.mppca",
        lambda data, **kwargs: (data, np.ones(data.shape[:3])),
    )
    with pytest.raises(PreprocessError, match="BET.*mask creation failed"):
        run_denoise(context)


def _write_denoise_inputs(context: PreprocessContext) -> tuple[Path, Path]:
    context.denoise_dir.mkdir(parents=True)
    pa_image = nib.load(context.config.dwi_pa)
    ap_image = nib.load(context.config.b0_ap)
    pa = np.asanyarray(pa_image.dataobj).astype(np.float32)
    ap = np.asanyarray(ap_image.dataobj).astype(np.float32)
    if ap.ndim == 3:
        ap = ap[..., None]
    pa_path = context.denoise_dir / "denoised_PA.nii.gz"
    ap_path = context.denoise_dir / "denoised_AP.nii.gz"
    _save_image(pa_path, pa, pa_image.affine)
    _save_image(ap_path, ap, ap_image.affine)
    return pa_path, ap_path


def test_run_gibbs_uses_configured_axis_and_preserves_denoised_sources(
    preprocess_context,
    monkeypatch,
):
    pa_path, ap_path = _write_denoise_inputs(preprocess_context)
    source_hashes = (sha256_file(pa_path), sha256_file(ap_path))
    calls = []

    def fake_gibbs(data, **kwargs):
        calls.append(kwargs.copy())
        data[...] = data + 1.0
        return data

    monkeypatch.setattr("dmri_pipeline.preprocess.gibbs_removal", fake_gibbs)
    details = run_gibbs(preprocess_context)

    assert [call["slice_axis"] for call in calls] == [2, 2]
    assert all(call["n_points"] == 3 for call in calls)
    assert all(call["inplace"] is True for call in calls)
    assert all(1 <= call["num_processes"] <= (preprocess_context.cpu_count or 1) for call in calls)
    assert (sha256_file(pa_path), sha256_file(ap_path)) == source_hashes
    assert nib.load(details.corrected_pa).get_data_dtype() == np.dtype(np.float32)
    assert nib.load(details.corrected_ap).get_data_dtype() == np.dtype(np.float32)


def test_run_gibbs_rejects_nonfinite_output(preprocess_context, monkeypatch):
    _write_denoise_inputs(preprocess_context)

    def fake_gibbs(data, **kwargs):
        data.flat[0] = np.inf
        return data

    monkeypatch.setattr("dmri_pipeline.preprocess.gibbs_removal", fake_gibbs)
    with pytest.raises(PreprocessError, match="finite"):
        run_gibbs(preprocess_context)


def test_run_gibbs_real_numerical_smoke_produces_finite_outputs(
    preprocess_context,
    monkeypatch,
):
    _write_denoise_inputs(preprocess_context)
    monkeypatch.setattr("dmri_pipeline.preprocess.os.cpu_count", lambda: 1)

    details = run_gibbs(preprocess_context)

    assert np.isfinite(np.asanyarray(nib.load(details.corrected_pa).dataobj)).all()
    assert np.isfinite(np.asanyarray(nib.load(details.corrected_ap).dataobj)).all()
    assert details.process_count == 1


def _write_gibbs_inputs(context: PreprocessContext) -> None:
    context.gibbs_dir.mkdir(parents=True, exist_ok=True)
    pa_source = nib.load(context.config.dwi_pa)
    ap_source = nib.load(context.config.b0_ap)
    pa = np.empty(pa_source.shape, dtype=np.float32)
    for volume in range(pa.shape[3]):
        pa[..., volume] = 100 + volume
    ap_data = np.asanyarray(ap_source.dataobj)
    if ap_data.ndim == 3:
        ap_data = ap_data[..., None]
    ap = np.empty(ap_data.shape, dtype=np.float32)
    for volume in range(ap.shape[3]):
        ap[..., volume] = 200 + volume
    _save_image(context.gibbs_dir / "gibbs_PA.nii.gz", pa, pa_source.affine)
    _save_image(context.gibbs_dir / "gibbs_AP.nii.gz", ap, ap_source.affine)


def _read_rows(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _read_indices(path: Path) -> list[int]:
    return [int(value) for value in path.read_text(encoding="utf-8").split()]


def test_prepare_topup_preserves_pa_then_ap_order(preprocess_context):
    _write_gibbs_inputs(preprocess_context)
    details = prepare_topup_inputs(preprocess_context)

    assert details.pa_b0_count == 1
    assert details.ap_b0_count == 2
    assert _read_rows(details.acqparams_topup) == [
        "0 -1 0 0.08",
        "0 1 0 0.08",
        "0 1 0 0.08",
    ]
    assert _read_indices(details.index_eddy) == [1] * 8
    merged = np.asanyarray(nib.load(details.merged_b0).dataobj)
    np.testing.assert_array_equal(merged[0, 0, 0], [100, 200, 201])


def test_prepare_topup_keeps_all_pa_and_ap_b0s_in_order(
    tmp_path,
    subject_config,
    fake_bet_executor,
):
    bvals = np.array([0, 40, 500, 1000, 1000, 2000, 3000, 3000])
    bvecs = np.loadtxt(subject_config.bvecs)
    bvecs[:, 1] = 0
    np.savetxt(subject_config.bvals, bvals[None, :], fmt="%.0f")
    np.savetxt(subject_config.bvecs, bvecs, fmt="%.8f")
    context = _make_context(
        tmp_path,
        subject_config,
        command_executor=fake_bet_executor,
    )
    _write_gibbs_inputs(context)

    details = prepare_topup_inputs(context)

    assert (details.pa_b0_count, details.ap_b0_count) == (2, 2)
    merged = np.asanyarray(nib.load(details.merged_b0).dataobj)
    np.testing.assert_array_equal(merged[0, 0, 0], [100, 101, 200, 201])
    assert _read_rows(details.acqparams_topup)[:2] == [
        "0 -1 0 0.08",
        "0 -1 0 0.08",
    ]


def test_prepare_topup_uses_nondefault_vectors_and_roundtrippable_readout(
    preprocess_context,
):
    acquisition = AcquisitionConfig(
        pa_vector=(-1, 0, 0),
        ap_vector=(1, 0, 0),
        total_readout_time=0.123456789012345,
        slice_axis=1,
    )
    context = replace(
        preprocess_context,
        config=replace(preprocess_context.config, acquisition=acquisition),
    )
    _write_gibbs_inputs(context)

    details = prepare_topup_inputs(context)

    assert _read_rows(details.acqparams_topup) == [
        "-1 0 0 0.123456789012345",
        "1 0 0 0.123456789012345",
        "1 0 0 0.123456789012345",
    ]
    assert _read_rows(details.acqparams_eddy) == [
        "-1 0 0 0.123456789012345",
        "1 0 0 0.123456789012345",
    ]
    assert float(_read_rows(details.acqparams_eddy)[0].split()[-1]) == (
        acquisition.total_readout_time
    )


def test_prepare_topup_writes_rounded_bvals_and_manifest(preprocess_context):
    np.savetxt(
        preprocess_context.config.bvals,
        np.array([[0, 249, 251, 1049, 1051, 2000, 2999, 3001]]),
        fmt="%.0f",
    )
    _write_gibbs_inputs(preprocess_context)

    details = prepare_topup_inputs(preprocess_context)

    assert _read_indices(details.bvals_rounded) == [
        0,
        200,
        300,
        1000,
        1100,
        2000,
        3000,
        3000,
    ]
    manifest = json.loads(details.manifest.read_text(encoding="utf-8"))
    assert manifest["volume_order"] == ["PA", "AP", "AP"]
    assert manifest["pa_b0_indices"] == [0]
    assert manifest["eddy_acquisition_row_order"] == ["PA", "AP"]


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("mismatch", "count"),
        ("empty", "b-values"),
        ("malformed", "b-values"),
    ],
)
def test_prepare_topup_rejects_bad_bvalues_actionably(
    preprocess_context,
    kind,
    message,
):
    _write_gibbs_inputs(preprocess_context)
    if kind == "mismatch":
        preprocess_context.config.bvals.write_text("0 1000\n", encoding="utf-8")
    elif kind == "empty":
        preprocess_context.config.bvals.write_text("", encoding="utf-8")
    else:
        preprocess_context.config.bvals.write_text("0 not-a-number\n", encoding="utf-8")

    with pytest.raises(PreprocessError, match=message):
        prepare_topup_inputs(preprocess_context)


def test_prepare_topup_refuses_nonfinite_or_mismatched_corrected_images(
    preprocess_context,
):
    _write_gibbs_inputs(preprocess_context)
    path = preprocess_context.gibbs_dir / "gibbs_AP.nii.gz"
    image = nib.load(path)
    data = np.asanyarray(image.dataobj).astype(np.float32)
    data.flat[0] = np.nan
    _save_image(path, data, image.affine)
    with pytest.raises(PreprocessError, match="finite"):
        prepare_topup_inputs(preprocess_context)


def test_prepare_topup_rejects_corrected_volume_counts_that_disagree_with_audit(
    preprocess_context,
):
    _write_gibbs_inputs(preprocess_context)
    path = preprocess_context.gibbs_dir / "gibbs_AP.nii.gz"
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)[..., :1]
    _save_image(path, data, image.affine)

    with pytest.raises(PreprocessError, match="audit"):
        prepare_topup_inputs(preprocess_context)


def test_clean_bet_mask_keeps_largest_component_and_records_metrics(
    preprocess_context,
):
    preprocess_context.bet_dir.mkdir(parents=True)
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[1:4, 1:4, 1:4] = 1
    mask[6, 6, 6] = 1
    _save_image(preprocess_context.bet_mask_source, mask, affine)

    details = clean_bet_mask(preprocess_context)

    cleaned_image = nib.load(details.cleaned_mask)
    cleaned = np.asanyarray(cleaned_image.dataobj)
    assert cleaned_image.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_array_equal(cleaned_image.affine, affine)
    assert int(cleaned.sum()) == 27
    assert cleaned[6, 6, 6] == 0
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics == {
        "component_count": 2,
        "largest_voxel_count": 27,
        "largest_tie": False,
        "original_voxel_count": 28,
        "removed_voxel_count": 1,
        "selected_component": 1,
    }


def test_clean_bet_mask_breaks_largest_ties_by_first_component(
    preprocess_context,
):
    preprocess_context.bet_dir.mkdir(parents=True)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1, 1, 1:3] = 1
    mask[6, 6, 5:7] = 1
    _save_image(preprocess_context.bet_mask_source, mask, np.eye(4))

    details = clean_bet_mask(preprocess_context)

    cleaned = np.asanyarray(nib.load(details.cleaned_mask).dataobj)
    np.testing.assert_array_equal(cleaned[1, 1, 1:3], [1, 1])
    np.testing.assert_array_equal(cleaned[6, 6, 5:7], [0, 0])
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics["largest_tie"] is True
    assert metrics["selected_component"] == 1


@pytest.mark.parametrize("case", ["empty", "four_dimensional", "nonfinite"])
def test_clean_bet_mask_rejects_invalid_masks(preprocess_context, case):
    preprocess_context.bet_dir.mkdir(parents=True)
    if case == "empty":
        data = np.zeros((8, 8, 8), dtype=np.float32)
        message = "foreground"
    elif case == "four_dimensional":
        data = np.ones((8, 8, 8, 1), dtype=np.float32)
        message = "3D"
    else:
        data = np.ones((8, 8, 8), dtype=np.float32)
        data.flat[0] = np.nan
        message = "finite"
    _save_image(preprocess_context.bet_mask_source, data, np.eye(4))

    with pytest.raises(PreprocessError, match=message):
        clean_bet_mask(preprocess_context)
