from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dmri_pipeline.models import (
    ModelContext,
    ModelInputError,
    ModelOutputError,
    fit_direct_dki,
    fit_dki,
    fit_dti,
    load_model_inputs,
    select_dti_volumes,
    validate_scalar_map,
    validate_vector_map,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directions(count: int) -> np.ndarray:
    """Return deterministic, non-collinear unit vectors."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    phi = golden_angle * indices
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


@pytest.fixture
def reference_image() -> nib.Nifti1Image:
    affine = np.array(
        [
            [1.7, 0.0, 0.0, 3.0],
            [0.0, 1.8, 0.0, -2.0],
            [0.0, 0.0, 1.9, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine)


def write_map(
    directory: Path,
    reference: nib.Nifti1Image,
    *,
    value: float,
    dtype: np.dtype | type = np.float32,
) -> Path:
    path = directory / "map.nii.gz"
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(
            np.full(reference.shape[:3], value, dtype=dtype),
            reference.affine,
            header,
        ),
        path,
    )
    return path


@pytest.fixture
def model_context(tmp_path: Path) -> ModelContext:
    inputs = tmp_path / "upstream"
    inputs.mkdir()
    shape = (2, 2, 2)
    bvals = np.array([0.0] + [1000.0] * 7 + [2000.0] * 8 + [3000.0] * 8)
    bvecs = np.zeros((3, bvals.size), dtype=float)
    bvecs[:, 1:] = _directions(bvals.size - 1).T
    spatial = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data = np.stack(
        [1000.0 - spatial * 2.0 - index * 3.0 for index in range(bvals.size)],
        axis=-1,
    ).astype(np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    mask.reshape(-1)[:5] = 1.0
    affine = np.diag([1.7, 1.8, 1.9, 1.0])

    dwi = inputs / "eddy_corrected.nii.gz"
    mask_path = inputs / "brain_mask.nii.gz"
    bval_path = inputs / "bvals"
    rotated = inputs / "eddy_rotated_bvecs"
    nib.save(nib.Nifti1Image(data, affine), dwi)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    np.savetxt(bval_path, bvals[None, :], fmt="%.1f")
    np.savetxt(rotated, bvecs, fmt="%.12f")

    package_root = Path(__file__).parents[1]
    return ModelContext(
        eddy_dwi=dwi,
        brain_mask=mask_path,
        bvals=bval_path,
        rotated_bvecs=rotated,
        work_dir=tmp_path / "model_work",
        henrique_helper=(
            package_root / "vendor" / "henrique_helpers" / "dki_alternative.py"
        ),
        dti_max_b=1200.0,
    )


def test_dti_selection_uses_configured_upper_b():
    bvals = np.array([0, 200, 500, 1000, 1200, 1200.1, 2000, 3000], dtype=float)
    selected = select_dti_volumes(bvals, max_b=1200)
    assert bvals[selected].max() <= 1200
    assert selected.sum() == 5
    np.testing.assert_array_equal(
        selected, [True, True, True, True, True, False, False, False]
    )


@pytest.mark.parametrize("max_b", [0, -1, np.inf, np.nan, True, "1200"])
def test_dti_selection_rejects_invalid_max_b(max_b):
    with pytest.raises(ModelInputError, match="max_b"):
        select_dti_volumes(np.array([0.0, 1000.0]), max_b=max_b)


def test_dti_selection_rejects_invalid_bvals():
    with pytest.raises(ModelInputError, match="b-values"):
        select_dti_volumes(np.array([0.0, np.nan]), max_b=1200)


def test_load_model_inputs_uses_rotated_bvecs_and_ignores_scanner_file(
    model_context: ModelContext,
):
    scanner_bvecs = model_context.rotated_bvecs.with_name("scanner_original.bvec")
    scanner_bvecs.write_text("intentionally invalid\n", encoding="utf-8")
    loaded = load_model_inputs(model_context)
    np.testing.assert_allclose(loaded.bvecs, np.loadtxt(model_context.rotated_bvecs))
    assert loaded.data.dtype == np.float32
    assert loaded.mask.dtype == np.bool_
    assert not loaded.data.flags.writeable
    assert not loaded.mask.flags.writeable
    assert not loaded.bvals.flags.writeable
    assert not loaded.bvecs.flags.writeable


def test_load_model_inputs_normalizes_n_by_3_rotated_bvecs(
    model_context: ModelContext,
):
    np.savetxt(
        model_context.rotated_bvecs,
        np.loadtxt(model_context.rotated_bvecs).T,
        fmt="%.12f",
    )
    loaded = load_model_inputs(model_context)
    assert loaded.bvecs.shape == (3, loaded.bvals.size)


def test_load_model_inputs_accepts_unit_length_b0_vector(
    model_context: ModelContext,
):
    bvecs = np.loadtxt(model_context.rotated_bvecs)
    bvecs[:, 0] = (1.0, 0.0, 0.0)
    np.savetxt(model_context.rotated_bvecs, bvecs, fmt="%.12f")

    loaded = load_model_inputs(model_context)

    np.testing.assert_array_equal(loaded.bvecs[:, 0], (1.0, 0.0, 0.0))


def test_load_model_inputs_accepts_mixed_zero_and_unit_b0_vectors(
    model_context: ModelContext,
):
    bvals = np.loadtxt(model_context.bvals).reshape(-1)
    bvals[1] = 0.0
    np.savetxt(model_context.bvals, bvals[None, :], fmt="%.1f")
    bvecs = np.loadtxt(model_context.rotated_bvecs)
    bvecs[:, 0] = 0.0
    bvecs[:, 1] = (1.0, 0.0, 0.0)
    np.savetxt(model_context.rotated_bvecs, bvecs, fmt="%.12f")

    loaded = load_model_inputs(model_context)

    np.testing.assert_array_equal(loaded.bvecs[:, :2], [[0, 1], [0, 0], [0, 0]])


def test_load_model_inputs_accepts_synthetic_177_volumes_with_11_unit_b0s_without_rewriting(
    model_context: ModelContext,
):
    volume_count = 177
    b0_count = 11
    image = nib.load(model_context.eddy_dwi)
    spatial = np.arange(np.prod(image.shape[:3]), dtype=np.float32).reshape(
        image.shape[:3]
    )
    data = np.stack(
        [1000.0 - spatial - index for index in range(volume_count)], axis=-1
    ).astype(np.float32)
    nib.save(nib.Nifti1Image(data, image.affine), model_context.eddy_dwi)
    bvals = np.concatenate(
        (
            np.zeros(b0_count),
            np.full(56, 1000.0),
            np.full(55, 2000.0),
            np.full(55, 3000.0),
        )
    )
    bvecs = np.empty((3, volume_count), dtype=float)
    bvecs[:, :b0_count] = np.array([[1.0], [0.0], [0.0]])
    bvecs[:, b0_count:] = _directions(volume_count - b0_count).T
    np.savetxt(model_context.bvals, bvals[None, :], fmt="%.1f")
    np.savetxt(model_context.rotated_bvecs, bvecs, fmt="%.12f")
    raw_paths = (
        model_context.eddy_dwi,
        model_context.brain_mask,
        model_context.bvals,
        model_context.rotated_bvecs,
    )
    before = {path: path.read_bytes() for path in raw_paths}

    loaded = load_model_inputs(model_context)

    assert loaded.data.shape == (2, 2, 2, 177)
    assert np.count_nonzero(loaded.bvals < 50) == 11
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dwi_3d", "4D"),
        ("dwi_float64", "float32"),
        ("dwi_nonfinite", "finite"),
        ("mask_4d", "3D"),
        ("mask_nonfinite", "finite"),
        ("mask_empty", "at least one voxel"),
        ("mask_shape", "same spatial shape"),
        ("mask_affine", "same image grid"),
        ("bvals_negative", "nonnegative"),
        ("bvals_nonfinite", "finite"),
        ("bvals_count", "count"),
        ("bvecs_count", "count"),
        ("b0_vector", "b0 vectors"),
        ("dw_vector", "unit length"),
    ],
)
def test_load_model_inputs_rejects_invalid_inputs(
    model_context: ModelContext,
    mutation: str,
    message: str,
):
    dwi_image = nib.load(model_context.eddy_dwi)
    dwi = np.asarray(dwi_image.dataobj)
    mask_image = nib.load(model_context.brain_mask)
    mask = np.asarray(mask_image.dataobj)
    bvals = np.loadtxt(model_context.bvals).reshape(-1)
    bvecs = np.loadtxt(model_context.rotated_bvecs)

    if mutation == "dwi_3d":
        nib.save(nib.Nifti1Image(dwi[..., 0], dwi_image.affine), model_context.eddy_dwi)
    elif mutation == "dwi_float64":
        nib.save(nib.Nifti1Image(dwi.astype(np.float64), dwi_image.affine), model_context.eddy_dwi)
    elif mutation == "dwi_nonfinite":
        dwi = dwi.copy()
        dwi.reshape(-1)[0] = np.nan
        nib.save(nib.Nifti1Image(dwi, dwi_image.affine), model_context.eddy_dwi)
    elif mutation == "mask_4d":
        nib.save(
            nib.Nifti1Image(mask[..., None], mask_image.affine), model_context.brain_mask
        )
    elif mutation == "mask_nonfinite":
        mask = mask.copy()
        mask.reshape(-1)[0] = np.nan
        nib.save(nib.Nifti1Image(mask, mask_image.affine), model_context.brain_mask)
    elif mutation == "mask_empty":
        nib.save(nib.Nifti1Image(np.zeros_like(mask), mask_image.affine), model_context.brain_mask)
    elif mutation == "mask_shape":
        nib.save(
            nib.Nifti1Image(np.ones((3, 2, 2), dtype=np.float32), mask_image.affine),
            model_context.brain_mask,
        )
    elif mutation == "mask_affine":
        affine = mask_image.affine.copy()
        affine[0, 3] += 0.01
        nib.save(nib.Nifti1Image(mask, affine), model_context.brain_mask)
    elif mutation == "bvals_negative":
        bvals[1] = -1
        np.savetxt(model_context.bvals, bvals[None, :])
    elif mutation == "bvals_nonfinite":
        bvals[1] = np.nan
        np.savetxt(model_context.bvals, bvals[None, :])
    elif mutation == "bvals_count":
        np.savetxt(model_context.bvals, bvals[:-1][None, :])
    elif mutation == "bvecs_count":
        np.savetxt(model_context.rotated_bvecs, bvecs[:, :-1])
    elif mutation == "b0_vector":
        bvecs[:, 0] = (0.5, 0.0, 0.0)
        np.savetxt(model_context.rotated_bvecs, bvecs)
    elif mutation == "dw_vector":
        bvecs[:, 1] *= 0.4
        np.savetxt(model_context.rotated_bvecs, bvecs)

    with pytest.raises(ModelInputError, match=message):
        load_model_inputs(model_context)


@pytest.mark.parametrize("norm", [0.0, 0.5])
def test_load_model_inputs_rejects_zero_or_intermediate_non_b0_vector(
    model_context: ModelContext,
    norm: float,
):
    bvecs = np.loadtxt(model_context.rotated_bvecs)
    bvecs[:, 1] = (norm, 0.0, 0.0)
    np.savetxt(model_context.rotated_bvecs, bvecs)

    with pytest.raises(ModelInputError, match="non-b0.*unit length"):
        load_model_inputs(model_context)


def test_load_model_inputs_rejects_too_few_noncollinear_directions(
    model_context: ModelContext,
):
    bvecs = np.loadtxt(model_context.rotated_bvecs)
    bvecs[:, 1:] = np.array([[1.0], [0.0], [0.0]])
    np.savetxt(model_context.rotated_bvecs, bvecs)
    with pytest.raises(ModelInputError, match="six.*non-collinear"):
        load_model_inputs(model_context)


def test_model_context_rejects_work_directory_that_aliases_upstream_parent(
    model_context: ModelContext,
):
    with pytest.raises(ModelInputError, match="work directory.*upstream"):
        replace(model_context, work_dir=model_context.eddy_dwi.parent)


def test_model_context_rejects_hardlinked_input_identities(
    model_context: ModelContext, tmp_path: Path
):
    helper_alias = tmp_path / "helper_alias.py"
    os.link(model_context.bvals, helper_alias)
    with pytest.raises(ModelInputError, match="same filesystem object"):
        replace(model_context, henrique_helper=helper_alias)


def test_map_validator_rejects_nonfinite_values(
    tmp_path: Path, reference_image: nib.Nifti1Image
):
    path = write_map(tmp_path, reference_image, value=np.nan)
    with pytest.raises(ModelOutputError, match="non-finite"):
        validate_scalar_map(path, reference_image.shape[:3])


def test_scalar_validator_rejects_shape_affine_and_nonfloat(
    tmp_path: Path, reference_image: nib.Nifti1Image
):
    wrong_shape = tmp_path / "wrong_shape.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2, 1), dtype=np.float32), reference_image.affine),
        wrong_shape,
    )
    with pytest.raises(ModelOutputError, match="3D"):
        validate_scalar_map(wrong_shape, reference_image.shape[:3])

    wrong_affine = write_map(tmp_path, reference_image, value=1.0)
    affine = reference_image.affine.copy()
    affine[0, 3] += 0.01
    nib.save(
        nib.Nifti1Image(np.ones(reference_image.shape, dtype=np.float32), affine),
        wrong_affine,
    )
    with pytest.raises(ModelOutputError, match="affine"):
        validate_scalar_map(
            wrong_affine, reference_image.shape[:3], reference_image.affine
        )

    integer_map = write_map(
        tmp_path, reference_image, value=1, dtype=np.int16
    )
    with pytest.raises(ModelOutputError, match="floating"):
        validate_scalar_map(integer_map, reference_image.shape[:3])


def test_vector_validator_requires_three_float32_finite_components(
    tmp_path: Path, reference_image: nib.Nifti1Image
):
    valid = tmp_path / "v1.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.ones(reference_image.shape[:3] + (3,), dtype=np.float32),
            reference_image.affine,
        ),
        valid,
    )
    validation = validate_vector_map(
        valid, reference_image.shape[:3], reference_image.affine
    )
    assert validation.shape == reference_image.shape[:3] + (3,)
    assert validation.dtype == "float32"

    invalid = tmp_path / "bad_v1.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.ones(reference_image.shape[:3] + (2,), dtype=np.float32),
            reference_image.affine,
        ),
        invalid,
    )
    with pytest.raises(ModelOutputError, match="final dimension 3"):
        validate_vector_map(invalid, reference_image.shape[:3])


class _TensorFit:
    def __init__(self, shape: tuple[int, int, int], *, entirely_invalid: bool = False):
        base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0
        self.fa = base.copy()
        self.fa.reshape(-1)[1] = 1.5
        self.fa.reshape(-1)[4] = np.nan
        self.md = np.full(shape, np.nan if entirely_invalid else 0.001)
        self.ad = np.full(shape, 0.0015)
        self.rd = np.full(shape, 0.0007)
        self.evecs = np.zeros(shape + (3, 3), dtype=float)
        self.evecs[..., 0, 0] = 1.0


class _KurtosisFit(_TensorFit):
    def __init__(self, shape: tuple[int, int, int]):
        super().__init__(shape)
        self.bounds: list[tuple[str, int, int]] = []

    def mk(self, *, min_kurtosis: int, max_kurtosis: int):
        self.bounds.append(("MK", min_kurtosis, max_kurtosis))
        return np.full(self.fa.shape, 0.8)

    def ak(self, *, min_kurtosis: int, max_kurtosis: int):
        self.bounds.append(("AK", min_kurtosis, max_kurtosis))
        return np.full(self.fa.shape, 1.0)

    def rk(self, *, min_kurtosis: int, max_kurtosis: int):
        self.bounds.append(("RK", min_kurtosis, max_kurtosis))
        return np.full(self.fa.shape, 0.6)


def _fa_with_infinities(shape: tuple[int, int, int]) -> np.ndarray:
    values = np.array(
        [np.inf, -np.inf, 1.5, -0.25, 0.5, np.inf, -np.inf, 0.25],
        dtype=float,
    )
    return values.reshape(shape)


def _patch_gradient_table(monkeypatch: pytest.MonkeyPatch, models_module, calls: dict):
    def fake_gradient_table(*, bvals, bvecs, b0_threshold):
        calls["gradient_bvals"] = np.asarray(bvals).copy()
        calls["gradient_bvecs"] = np.asarray(bvecs).copy()
        calls["b0_threshold"] = b0_threshold
        return object()

    monkeypatch.setattr(models_module, "gradient_table", fake_gradient_table)


def test_dti_uses_nlls_selected_volumes_and_writes_valid_sanitized_maps(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    calls: dict[str, object] = {}
    _patch_gradient_table(monkeypatch, models, calls)

    class FakeTensorModel:
        def __init__(self, gtab, *, fit_method):
            calls["dti_method"] = fit_method

        def fit(self, data, *, mask):
            calls["dti_data_shape"] = data.shape
            calls["dti_mask"] = mask.copy()
            return _TensorFit(data.shape[:3])

    monkeypatch.setattr(models.dti, "TensorModel", FakeTensorModel)
    upstream_hashes = {
        path: _sha256(path)
        for path in (
            model_context.eddy_dwi,
            model_context.brain_mask,
            model_context.bvals,
            model_context.rotated_bvecs,
        )
    }

    details = fit_dti(model_context)

    assert calls["dti_method"] == "NLLS"
    assert calls["dti_data_shape"] == (2, 2, 2, 8)
    assert calls["b0_threshold"] == 50
    assert np.asarray(calls["gradient_bvals"]).size == 8
    assert set(details.maps) == {"FA", "MD", "AD", "RD", "V1"}
    for name, path in details.maps.items():
        image = nib.load(path)
        assert image.get_data_dtype() == np.dtype(np.float32)
        assert np.array_equal(image.affine, nib.load(model_context.eddy_dwi).affine)
        if name == "V1":
            assert image.shape == (2, 2, 2, 3)
            validate_vector_map(path, (2, 2, 2), image.affine)
        else:
            assert image.shape == (2, 2, 2)
            validate_scalar_map(path, (2, 2, 2), image.affine)
        saved = np.asarray(image.dataobj)
        assert np.all(saved.reshape((-1,) + saved.shape[3:])[5:] == 0)

    fa = np.asarray(nib.load(details.maps["FA"]).dataobj)
    assert fa.min() >= 0
    assert fa.max() <= 1
    assert fa.reshape(-1)[4] == 0

    metrics_text = details.metrics.read_text(encoding="utf-8")
    assert "NaN" not in metrics_text
    metrics = json.loads(metrics_text)
    assert metrics["model"] == "DIPY TensorModel NLLS"
    assert metrics["dti_max_b"] == 1200.0
    assert metrics["selected_volume_count"] == 8
    assert metrics["selected_indices_zero_based"] == list(range(8))
    assert metrics["selected_indices_one_based"] == list(range(1, 9))
    assert metrics["brain_voxel_count"] == 5
    assert metrics["nonfinite_replaced"]["FA"] == {
        "inside_mask": 1,
        "outside_mask": 0,
        "total": 1,
    }
    for path, digest in upstream_hashes.items():
        assert _sha256(path) == digest


def test_dti_fails_when_a_requested_map_is_entirely_invalid_inside_mask(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    _patch_gradient_table(monkeypatch, models, {})

    class InvalidTensorModel:
        def __init__(self, gtab, *, fit_method):
            pass

        def fit(self, data, *, mask):
            return _TensorFit(data.shape[:3], entirely_invalid=True)

    monkeypatch.setattr(models.dti, "TensorModel", InvalidTensorModel)
    with pytest.raises(ModelOutputError, match="MD.*no finite.*brain mask"):
        fit_dti(model_context)


def test_dti_counts_infinite_fa_before_clipping_and_clips_finite_values(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    _patch_gradient_table(monkeypatch, models, {})

    class InfiniteFATensorModel:
        def __init__(self, gtab, *, fit_method):
            pass

        def fit(self, data, *, mask):
            fit = _TensorFit(data.shape[:3])
            fit.fa = _fa_with_infinities(data.shape[:3])
            return fit

    monkeypatch.setattr(models.dti, "TensorModel", InfiniteFATensorModel)
    details = fit_dti(model_context)
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics["nonfinite_replaced"]["FA"] == {
        "inside_mask": 2,
        "outside_mask": 2,
        "total": 4,
    }
    fa = np.asarray(nib.load(details.maps["FA"]).dataobj).reshape(-1)
    np.testing.assert_array_equal(fa, [0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])


def test_dti_fails_gradient_preflight_before_constructing_dipy_model(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    monkeypatch.setattr(
        models.dti,
        "TensorModel",
        lambda *args, **kwargs: pytest.fail("DIPY constructor was reached"),
    )
    with pytest.raises(ModelInputError, match="DTI selection.*six.*non-collinear"):
        fit_dti(replace(model_context, dti_max_b=500.0))


def test_dti_rejects_rank_deficient_selected_design_before_dipy(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    bvecs = np.loadtxt(model_context.rotated_bvecs)
    angles = np.linspace(0.0, np.pi, 7, endpoint=False)
    bvecs[:, 1:8] = np.vstack(
        (np.cos(angles), np.sin(angles), np.zeros(angles.size))
    )
    np.savetxt(model_context.rotated_bvecs, bvecs)
    monkeypatch.setattr(
        models.dti,
        "TensorModel",
        lambda *args, **kwargs: pytest.fail("DIPY constructor was reached"),
    )
    with pytest.raises(ModelInputError, match="DTI design matrix.*rank"):
        fit_dti(model_context)


def test_fit_rejects_preexisting_output_hardlink_without_mutating_upstream(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    calls: dict[str, object] = {}
    _patch_gradient_table(monkeypatch, models, calls)

    class FakeTensorModel:
        def __init__(self, gtab, *, fit_method):
            pass

        def fit(self, data, *, mask):
            return _TensorFit(data.shape[:3])

    monkeypatch.setattr(models.dti, "TensorModel", FakeTensorModel)
    model_context.work_dir.mkdir()
    output_alias = model_context.work_dir / "FA.nii.gz"
    os.link(model_context.eddy_dwi, output_alias)
    before = _sha256(model_context.eddy_dwi)
    with pytest.raises(ModelOutputError, match="pre-existing outputs"):
        fit_dti(model_context)
    assert _sha256(model_context.eddy_dwi) == before
    assert os.path.samefile(output_alias, model_context.eddy_dwi)


def test_standard_dki_uses_wls_all_volumes_and_bounded_kurtosis(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    calls: dict[str, object] = {}
    fit_holder: dict[str, _KurtosisFit] = {}
    _patch_gradient_table(monkeypatch, models, calls)

    class FakeDKIModel:
        def __init__(self, gtab, *, fit_method):
            calls["dki_method"] = fit_method

        def fit(self, data, *, mask):
            calls["dki_data_shape"] = data.shape
            fit_holder["fit"] = _KurtosisFit(data.shape[:3])
            return fit_holder["fit"]

    monkeypatch.setattr(models.dki, "DiffusionKurtosisModel", FakeDKIModel)
    details = fit_dki(replace(model_context, work_dir=model_context.work_dir / "dki"))

    assert calls["dki_method"] == "WLS"
    assert calls["dki_data_shape"] == (2, 2, 2, 24)
    assert np.asarray(calls["gradient_bvals"]).size == 24
    assert fit_holder["fit"].bounds == [
        ("MK", -1, 3),
        ("AK", -1, 3),
        ("RK", -1, 3),
    ]
    assert set(details.maps) == {"FA", "MD", "AD", "RD", "V1", "MK", "AK", "RK"}
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics["model"] == "DIPY DiffusionKurtosisModel WLS"
    assert metrics["selection"] == "all acquired shells"
    assert metrics["selected_volume_count"] == 24


def test_standard_dki_fails_gradient_preflight_before_dipy(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    bvecs = np.loadtxt(model_context.rotated_bvecs)
    six_axes = _directions(6).T
    for index in range(1, bvecs.shape[1]):
        bvecs[:, index] = six_axes[:, (index - 1) % 6]
    np.savetxt(model_context.rotated_bvecs, bvecs)
    monkeypatch.setattr(
        models.dki,
        "DiffusionKurtosisModel",
        lambda *args, **kwargs: pytest.fail("DIPY constructor was reached"),
    )
    with pytest.raises(ModelInputError, match="standard DKI.*15.*non-collinear"):
        fit_dki(model_context)


def test_dki_counts_infinite_fa_before_clipping_and_clips_finite_values(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    _patch_gradient_table(monkeypatch, models, {})

    class InfiniteFADKIModel:
        def __init__(self, gtab, *, fit_method):
            pass

        def fit(self, data, *, mask):
            fit = _KurtosisFit(data.shape[:3])
            fit.fa = _fa_with_infinities(data.shape[:3])
            return fit

    monkeypatch.setattr(models.dki, "DiffusionKurtosisModel", InfiniteFADKIModel)
    details = fit_dki(replace(model_context, work_dir=model_context.work_dir / "dki"))
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics["nonfinite_replaced"]["FA"] == {
        "inside_mask": 2,
        "outside_mask": 2,
        "total": 4,
    }
    fa = np.asarray(nib.load(details.maps["FA"]).dataobj).reshape(-1)
    np.testing.assert_array_equal(fa, [0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])


def test_standard_dki_rejects_rank_deficient_design_before_dipy(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    bvecs = np.loadtxt(model_context.rotated_bvecs)
    angles = np.linspace(0.0, np.pi, bvecs.shape[1] - 1, endpoint=False)
    bvecs[:, 1:] = np.vstack(
        (np.cos(angles), np.sin(angles), np.zeros(angles.size))
    )
    np.savetxt(model_context.rotated_bvecs, bvecs)
    monkeypatch.setattr(
        models.dki,
        "DiffusionKurtosisModel",
        lambda *args, **kwargs: pytest.fail("DIPY constructor was reached"),
    )
    with pytest.raises(ModelInputError, match="DKI design matrix.*rank"):
        fit_dki(model_context)


def test_direct_dki_invokes_vendored_function_with_all_volumes_and_parameter_order(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    calls: dict[str, object] = {}
    _patch_gradient_table(monkeypatch, models, calls)

    def fake_avs(gtab, data, *, mask):
        calls["direct_data_shape"] = data.shape
        calls["direct_mask"] = mask.copy()
        params = np.zeros(data.shape[:3] + (3,), dtype=float)
        params[..., 0] = 0.001
        params[..., 1] = 0.9
        params[..., 2] = 1000.0
        return params

    monkeypatch.setattr(models, "_load_avs_dki_df", lambda path: fake_avs)
    details = fit_direct_dki(
        replace(model_context, work_dir=model_context.work_dir / "direct")
    )

    assert calls["direct_data_shape"] == (2, 2, 2, 24)
    assert np.asarray(calls["gradient_bvals"]).size == 24
    assert set(details.maps) == {"MD", "MK", "S0"}
    for name, expected in (("MD", 0.001), ("MK", 0.9), ("S0", 1000.0)):
        values = np.asarray(nib.load(details.maps[name]).dataobj)
        np.testing.assert_allclose(values.reshape(-1)[:5], expected)
        np.testing.assert_array_equal(values.reshape(-1)[5:], 0)
    metrics = json.loads(details.metrics.read_text(encoding="utf-8"))
    assert metrics["model"] == "Henrique avs_dki_df average-signal direct fit"
    assert metrics["selection"] == "all acquired shells"
    assert metrics["helper_sha256"] == _sha256(model_context.henrique_helper)


def test_direct_dki_fails_shell_preflight_before_loading_helper(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    bvals = np.loadtxt(model_context.bvals).reshape(-1)
    bvals[bvals >= 50] = 1000
    np.savetxt(model_context.bvals, bvals[None, :])
    monkeypatch.setattr(
        models,
        "_load_avs_dki_df",
        lambda path: pytest.fail("Henrique helper was loaded"),
    )
    with pytest.raises(ModelInputError, match="direct DKI.*three acquired shells"):
        fit_direct_dki(model_context)


def test_direct_dki_rejects_helper_quantized_rank_deficient_shell_design(
    model_context: ModelContext, monkeypatch: pytest.MonkeyPatch
):
    import dmri_pipeline.models as models

    bvals = np.array([0.0] + [9500.0] * 11 + [10000.0] * 12)
    np.savetxt(model_context.bvals, bvals[None, :])
    monkeypatch.setattr(
        models,
        "_load_avs_dki_df",
        lambda path: pytest.fail("Henrique helper was loaded"),
    )
    with pytest.raises(
        ModelInputError,
        match=r"Henrique direct DKI shell design has rank 2; full rank 3 is required",
    ):
        fit_direct_dki(model_context)


def test_vendored_helper_has_fixed_archived_identity():
    package_root = Path(__file__).parents[1]
    vendored = package_root / "vendor" / "henrique_helpers" / "dki_alternative.py"
    assert _sha256(vendored) == (
        "f046fa5c3bdff397cd44562c40de1c5aad19de591f10d8b13e23de73b6371490"
    )


def test_vendored_helper_imports_and_runs_under_pinned_dipy_without_sys_path_change():
    from dipy.core.gradients import gradient_table

    import dmri_pipeline.models as models

    package_root = Path(__file__).parents[1]
    helper = package_root / "vendor" / "henrique_helpers" / "dki_alternative.py"
    before = tuple(sys.path)
    avs_dki_df = models._load_avs_dki_df(helper)
    bvals = np.array([0.0, 1000.0, 2000.0])
    bvecs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gtab = gradient_table(bvals=bvals, bvecs=bvecs, b0_threshold=50)
    params = avs_dki_df(
        gtab,
        np.array([[[[1000.0, 600.0, 400.0]]]], dtype=np.float32),
        mask=np.ones((1, 1, 1), dtype=bool),
    )
    assert params.shape == (1, 1, 1, 3)
    assert np.isfinite(params).all()
    assert tuple(sys.path) == before
