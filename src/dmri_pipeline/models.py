"""Validated DTI, standard DKI, and direct average-signal DKI fits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import types
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst import dki, dti
from dipy.reconst.utils import dki_design_matrix
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError, ImageDataError

from .utils import InputAuditError, normalize_bvecs, round_shells


class ModelInputError(ValueError):
    """Raised before fitting when model inputs violate the scientific contract."""


class ModelOutputError(RuntimeError):
    """Raised when a requested model map cannot be safely stored or validated."""


@dataclass(frozen=True)
class ModelContext:
    """Explicit upstream inputs and one assigned model-stage work directory."""

    eddy_dwi: Path
    brain_mask: Path
    bvals: Path
    rotated_bvecs: Path
    work_dir: Path
    henrique_helper: Path
    dti_max_b: float = 1200.0

    def __post_init__(self) -> None:
        path_fields = (
            "eddy_dwi",
            "brain_mask",
            "bvals",
            "rotated_bvecs",
            "work_dir",
            "henrique_helper",
        )
        for name in path_fields:
            if not isinstance(getattr(self, name), Path):
                raise ModelInputError(f"{name} must be a pathlib.Path")
        if (
            isinstance(self.dti_max_b, bool)
            or not isinstance(self.dti_max_b, Real)
            or not math.isfinite(float(self.dti_max_b))
            or float(self.dti_max_b) <= 0
        ):
            raise ModelInputError("dti_max_b must be a positive finite number")
        input_paths = (
            self.eddy_dwi,
            self.brain_mask,
            self.bvals,
            self.rotated_bvecs,
            self.henrique_helper,
        )
        _validate_context_identities(self, input_paths)


@dataclass(frozen=True)
class ModelInputs:
    """Read-only in-memory model data plus a copied reference geometry."""

    data: np.ndarray
    mask: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    reference_affine: np.ndarray
    reference_header: nib.nifti1.Nifti1Header

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.data.shape[:3])


@dataclass(frozen=True)
class MapValidation:
    """Validated shape and stored dtype for one model output."""

    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ModelFitDetails:
    """Materialized maps and deterministic metrics for one model branch."""

    maps: Mapping[str, Path]
    metrics: Path


def load_model_inputs(context: ModelContext) -> ModelInputs:
    """Load and validate EDDY data, mask, b-values, and rotated b-vectors."""
    if not isinstance(context, ModelContext):
        raise ModelInputError("context must be a ModelContext")
    _validate_context_identities(
        context,
        (
            context.eddy_dwi,
            context.brain_mask,
            context.bvals,
            context.rotated_bvecs,
            context.henrique_helper,
        ),
    )

    dwi_image = _load_image(context.eddy_dwi, "EDDY DWI")
    if len(dwi_image.shape) != 4:
        raise ModelInputError("EDDY DWI must be 4D")
    if np.dtype(dwi_image.get_data_dtype()) != np.dtype(np.float32):
        raise ModelInputError("EDDY DWI must have stored float32 data")
    data = _image_array(dwi_image, "EDDY DWI", dtype=np.float32)
    if not np.isfinite(data).all():
        raise ModelInputError("EDDY DWI data must be finite")

    mask_image = _load_image(context.brain_mask, "brain mask")
    if len(mask_image.shape) != 3:
        raise ModelInputError("brain mask must be 3D")
    mask_values = _image_array(mask_image, "brain mask")
    if not np.issubdtype(mask_values.dtype, np.number):
        raise ModelInputError("brain mask data must be numeric")
    if not np.isfinite(mask_values).all():
        raise ModelInputError("brain mask data must be finite")
    if tuple(mask_image.shape) != tuple(dwi_image.shape[:3]):
        raise ModelInputError("EDDY DWI and brain mask must have the same spatial shape")
    if not np.allclose(dwi_image.affine, mask_image.affine, atol=1e-5, rtol=0.0):
        raise ModelInputError("EDDY DWI and brain mask must use the same image grid")
    mask = np.asarray(mask_values > 0, dtype=bool)
    if not mask.any():
        raise ModelInputError("brain mask must contain at least one voxel")

    bvals = _load_text(context.bvals, "b-values").reshape(-1)
    if not np.isfinite(bvals).all():
        raise ModelInputError("b-values must be finite")
    if np.any(bvals < 0):
        raise ModelInputError("b-values must be nonnegative")
    if bvals.size != data.shape[-1]:
        raise ModelInputError("b-value count must equal EDDY DWI volume count")

    try:
        bvecs = normalize_bvecs(
            _load_text(context.rotated_bvecs, "EDDY-rotated b-vectors"),
            int(data.shape[-1]),
        )
    except InputAuditError as error:
        raise ModelInputError(str(error)) from error
    b0_mask = bvals < 50.0
    if not b0_mask.any():
        raise ModelInputError("model inputs must contain at least one b0 with b < 50")
    norms = np.linalg.norm(bvecs, axis=0)
    accepted_b0 = (norms[b0_mask] < 0.1) | (
        (norms[b0_mask] >= 0.95) & (norms[b0_mask] <= 1.05)
    )
    if not accepted_b0.all():
        raise ModelInputError("b0 vectors must be near-zero or unit length")
    diffusion = ~b0_mask
    if np.any((norms[diffusion] < 0.95) | (norms[diffusion] > 1.05)):
        raise ModelInputError(
            "non-b0 EDDY-rotated b-vectors must have unit length "
            "(norm 0.95 to 1.05)"
        )
    if _count_gradient_axes(bvecs[:, diffusion]) < 6:
        raise ModelInputError(
            "model inputs need at least six non-collinear diffusion directions"
        )

    affine = np.array(dwi_image.affine, dtype=float, copy=True)
    if not np.isfinite(affine).all():
        raise ModelInputError("EDDY DWI affine must be finite")
    header = dwi_image.header.copy()
    arrays = (
        np.array(data, dtype=np.float32, copy=True),
        np.array(mask, dtype=bool, copy=True),
        np.array(bvals, dtype=float, copy=True),
        np.array(bvecs, dtype=float, copy=True),
        affine,
    )
    for array in arrays:
        array.setflags(write=False)
    return ModelInputs(*arrays[:4], arrays[4], header)


def select_dti_volumes(bvals: object, max_b: float) -> np.ndarray:
    """Select b0 and diffusion volumes at or below an inclusive DTI threshold."""
    if (
        isinstance(max_b, bool)
        or not isinstance(max_b, Real)
        or not math.isfinite(float(max_b))
        or float(max_b) <= 0
    ):
        raise ModelInputError("max_b must be a positive finite number")
    try:
        values = np.asarray(bvals, dtype=float)
    except (TypeError, ValueError) as error:
        raise ModelInputError("b-values must be a numeric one-dimensional array") from error
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ModelInputError("b-values must be finite, nonnegative, and one-dimensional")
    return values <= float(max_b)


def validate_scalar_map(
    path: Path,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray | None = None,
) -> MapValidation:
    """Load a saved scalar map and validate its geometry, dtype, and values."""
    image = _load_output_image(path)
    if len(image.shape) != 3:
        raise ModelOutputError(f"scalar map must be 3D: {path}")
    _validate_output_common(image, path, reference_shape, reference_affine)
    return MapValidation(Path(path), tuple(int(v) for v in image.shape), "float32")


def validate_vector_map(
    path: Path,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray | None = None,
) -> MapValidation:
    """Load a saved principal-eigenvector map and require a final dimension of 3."""
    image = _load_output_image(path)
    if len(image.shape) != 4 or image.shape[-1] != 3:
        raise ModelOutputError(f"vector map must be 4D with final dimension 3: {path}")
    _validate_output_common(image, path, reference_shape, reference_affine)
    return MapValidation(Path(path), tuple(int(v) for v in image.shape), "float32")


def fit_dti(context: ModelContext) -> ModelFitDetails:
    """Fit DIPY's NLLS diffusion tensor model to configured low-b volumes."""
    inputs = load_model_inputs(context)
    selected = select_dti_volumes(inputs.bvals, context.dti_max_b)
    _require_dti_gradients(inputs.bvals[selected], inputs.bvecs[:, selected])
    gtab = gradient_table(
        bvals=inputs.bvals[selected],
        bvecs=inputs.bvecs[:, selected].T,
        b0_threshold=50,
    )
    fit = dti.TensorModel(gtab, fit_method="NLLS").fit(
        inputs.data[..., selected], mask=inputs.mask
    )
    maps = {
        "FA": np.asarray(fit.fa),
        "MD": np.asarray(fit.md),
        "AD": np.asarray(fit.ad),
        "RD": np.asarray(fit.rd),
        "V1": np.asarray(fit.evecs)[..., :, 0],
    }
    indices = np.flatnonzero(selected)
    metrics: dict[str, object] = {
        "model": "DIPY TensorModel NLLS",
        "selection": f"b <= {float(context.dti_max_b):g} s/mm^2",
        "dti_max_b": float(context.dti_max_b),
        "selected_indices_zero_based": [int(value) for value in indices],
        "selected_indices_one_based": [int(value + 1) for value in indices],
        "selected_volume_count": int(selected.sum()),
        "selected_b0_count": int(np.count_nonzero(inputs.bvals[selected] < 50)),
        "brain_voxel_count": int(inputs.mask.sum()),
    }
    return _store_fit(context, inputs, maps, "dti_metrics.json", metrics, "DIPY DTI")


def fit_dki(context: ModelContext) -> ModelFitDetails:
    """Fit DIPY's WLS standard diffusion-kurtosis model using all volumes."""
    inputs = load_model_inputs(context)
    _require_dki_gradients(inputs)
    gtab = gradient_table(
        bvals=inputs.bvals,
        bvecs=inputs.bvecs.T,
        b0_threshold=50,
    )
    fit = dki.DiffusionKurtosisModel(gtab, fit_method="WLS").fit(
        inputs.data, mask=inputs.mask
    )
    maps = {
        "FA": np.asarray(fit.fa),
        "MD": np.asarray(fit.md),
        "AD": np.asarray(fit.ad),
        "RD": np.asarray(fit.rd),
        "V1": np.asarray(fit.evecs)[..., :, 0],
        "MK": np.asarray(fit.mk(min_kurtosis=-1, max_kurtosis=3)),
        "AK": np.asarray(fit.ak(min_kurtosis=-1, max_kurtosis=3)),
        "RK": np.asarray(fit.rk(min_kurtosis=-1, max_kurtosis=3)),
    }
    metrics: dict[str, object] = {
        "model": "DIPY DiffusionKurtosisModel WLS",
        "selection": "all acquired shells",
        "selected_volume_count": int(inputs.bvals.size),
        "shells": [int(value) for value in np.unique(round_shells(inputs.bvals))],
        "brain_voxel_count": int(inputs.mask.sum()),
        "kurtosis_bounds": {"minimum": -1, "maximum": 3},
    }
    return _store_fit(context, inputs, maps, "dki_metrics.json", metrics, "DIPY DKI")


def fit_direct_dki(context: ModelContext) -> ModelFitDetails:
    """Fit Henrique's direct average-signal DKI model using all volumes."""
    inputs = load_model_inputs(context)
    _require_direct_gradients(inputs)
    gtab = gradient_table(
        bvals=inputs.bvals,
        bvecs=inputs.bvecs.T,
        b0_threshold=50,
    )
    avs_dki_df = _load_avs_dki_df(context.henrique_helper)
    params = np.asarray(avs_dki_df(gtab, inputs.data, mask=inputs.mask))
    expected = inputs.spatial_shape + (3,)
    if params.shape != expected:
        raise ModelOutputError(
            f"Henrique avs_dki_df returned shape {params.shape}; expected {expected}"
        )
    maps = {"MD": params[..., 0], "MK": params[..., 1], "S0": params[..., 2]}
    metrics: dict[str, object] = {
        "model": "Henrique avs_dki_df average-signal direct fit",
        "selection": "all acquired shells",
        "selected_volume_count": int(inputs.bvals.size),
        "shells": [int(value) for value in np.unique(round_shells(inputs.bvals))],
        "brain_voxel_count": int(inputs.mask.sum()),
        "helper_sha256": _sha256(context.henrique_helper),
    }
    return _store_fit(
        context,
        inputs,
        maps,
        "dki_direct_metrics.json",
        metrics,
        "Henrique average-signal direct DKI",
    )


def _store_fit(
    context: ModelContext,
    inputs: ModelInputs,
    raw_maps: Mapping[str, np.ndarray],
    metrics_name: str,
    base_metrics: dict[str, object],
    description_prefix: str,
) -> ModelFitDetails:
    context.work_dir.mkdir(parents=True, exist_ok=True)
    if context.work_dir.is_symlink() or not context.work_dir.is_dir():
        raise ModelOutputError("model work directory must be a real directory")
    paths = {
        name: context.work_dir / f"{name}.nii.gz" for name in raw_maps
    }
    metrics_path = context.work_dir / metrics_name
    _require_absent((*paths.values(), metrics_path))

    sanitized: dict[str, np.ndarray] = {}
    nonfinite: dict[str, dict[str, int]] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    for name, values in raw_maps.items():
        is_vector = name == "V1"
        array, counts, summary = _sanitize_map(
            name, values, inputs.mask, inputs.spatial_shape, is_vector=is_vector
        )
        sanitized[name] = array
        nonfinite[name] = counts
        if not is_vector:
            summaries[name] = summary

    for name, array in sanitized.items():
        _save_float32(
            inputs,
            array,
            paths[name],
            f"{description_prefix} {name}",
        )
        if name == "V1":
            validate_vector_map(paths[name], inputs.spatial_shape, inputs.reference_affine)
        else:
            validate_scalar_map(paths[name], inputs.spatial_shape, inputs.reference_affine)

    metrics = dict(base_metrics)
    metrics["nonfinite_replaced"] = nonfinite
    metrics["map_summaries"] = summaries
    _write_json(metrics_path, metrics)
    return ModelFitDetails(MappingProxyType(paths), metrics_path)


def _sanitize_map(
    name: str,
    values: object,
    mask: np.ndarray,
    spatial_shape: tuple[int, int, int],
    *,
    is_vector: bool,
) -> tuple[np.ndarray, dict[str, int], dict[str, float | int]]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ModelOutputError(f"{name} map must be numeric") from error
    expected = spatial_shape + ((3,) if is_vector else ())
    if array.shape != expected:
        raise ModelOutputError(f"{name} map has shape {array.shape}; expected {expected}")
    expanded_mask = mask[..., None] if is_vector else mask
    expanded_mask = np.broadcast_to(expanded_mask, array.shape)
    finite = np.isfinite(array)
    finite_inside = finite & expanded_mask
    if not finite_inside.any():
        raise ModelOutputError(f"{name} map has no finite value inside the brain mask")
    inside = int(np.count_nonzero((~finite) & expanded_mask))
    outside = int(np.count_nonzero((~finite) & (~expanded_mask)))
    cleaned = np.array(array, dtype=np.float32, copy=True)
    cleaned[~finite] = 0.0
    cleaned[~expanded_mask] = 0.0
    if name == "FA":
        np.clip(cleaned, 0.0, 1.0, out=cleaned)
    inside_values = cleaned[finite_inside]
    summary = {
        "count": int(inside_values.size),
        "mean": float(np.mean(inside_values)),
        "median": float(np.median(inside_values)),
        "p01": float(np.percentile(inside_values, 1)),
        "p99": float(np.percentile(inside_values, 99)),
    }
    return (
        cleaned,
        {"inside_mask": inside, "outside_mask": outside, "total": inside + outside},
        summary,
    )


def _save_float32(
    inputs: ModelInputs, data: np.ndarray, path: Path, description: str
) -> None:
    header = inputs.reference_header.copy()
    header.set_data_dtype(np.float32)
    header["descrip"] = description[:79].encode("ascii", errors="ignore")
    try:
        nib.save(
            nib.Nifti1Image(
                np.asarray(data, dtype=np.float32),
                np.array(inputs.reference_affine, copy=True),
                header,
            ),
            path,
        )
    except (OSError, HeaderDataError, ImageDataError) as error:
        raise ModelOutputError(f"cannot write model map: {path}") from error


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        os.replace(temporary_name, path)
    except (OSError, ValueError) as error:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise ModelOutputError(f"cannot write model metrics: {path}") from error


def _validate_output_common(
    image: nib.spatialimages.SpatialImage,
    path: Path,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray | None,
) -> None:
    if tuple(int(value) for value in image.shape[:3]) != tuple(reference_shape):
        raise ModelOutputError(f"model map has the wrong spatial shape: {path}")
    if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        raise ModelOutputError(f"model map must use stored floating float32 data: {path}")
    if reference_affine is not None and not np.allclose(
        image.affine, np.asarray(reference_affine), atol=1e-5, rtol=0.0
    ):
        raise ModelOutputError(f"model map affine does not match the reference: {path}")
    values = _output_array(image, path)
    if not np.isfinite(values).all():
        raise ModelOutputError(f"model map contains non-finite values: {path}")


def _load_image(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(Path(path))
    except (OSError, ImageFileError, HeaderDataError, ImageDataError) as error:
        raise ModelInputError(f"cannot read {label}: {path}") from error
    if not np.isfinite(image.affine).all():
        raise ModelInputError(f"{label} affine must be finite")
    return image


def _load_output_image(path: Path) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(Path(path))
    except (OSError, ImageFileError, HeaderDataError, ImageDataError) as error:
        raise ModelOutputError(f"cannot read model map: {path}") from error
    return image


def _image_array(
    image: nib.spatialimages.SpatialImage,
    label: str,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    try:
        return np.asanyarray(image.dataobj, dtype=dtype)
    except (OSError, ValueError, TypeError, ImageDataError) as error:
        raise ModelInputError(f"cannot read {label} voxel data") from error


def _output_array(
    image: nib.spatialimages.SpatialImage, path: Path
) -> np.ndarray:
    try:
        return np.asanyarray(image.dataobj)
    except (OSError, ValueError, TypeError, ImageDataError) as error:
        raise ModelOutputError(f"cannot read model map voxel data: {path}") from error


def _load_text(path: Path, label: str) -> np.ndarray:
    try:
        return np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    except (OSError, ValueError) as error:
        raise ModelInputError(f"cannot read {label}: {path}") from error


def _count_gradient_axes(vectors: np.ndarray) -> int:
    axes: list[np.ndarray] = []
    for vector in np.asarray(vectors).T:
        axis = vector / np.linalg.norm(vector)
        first = np.flatnonzero(np.abs(axis) > 1e-8)
        if not first.size:
            continue
        if axis[first[0]] < 0:
            axis = -axis
        if not any(
            np.allclose(axis, existing, atol=1e-5, rtol=0.0) for existing in axes
        ):
            axes.append(axis)
    return len(axes)


def _require_dti_gradients(bvals: np.ndarray, bvecs: np.ndarray) -> None:
    if not np.any(bvals < 50):
        raise ModelInputError("DTI selection must include at least one b0")
    diffusion = bvals >= 50
    if _count_gradient_axes(bvecs[:, diffusion]) < 6:
        raise ModelInputError(
            "DTI selection needs at least six non-collinear diffusion directions"
        )
    x, y, z = bvecs
    design = np.column_stack(
        (
            bvals * x * x,
            2.0 * bvals * x * y,
            bvals * y * y,
            2.0 * bvals * x * z,
            2.0 * bvals * y * z,
            bvals * z * z,
            np.ones(bvals.size),
        )
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank < 7:
        raise ModelInputError(
            f"DTI design matrix has rank {rank}; full rank 7 is required"
        )


def _require_dki_gradients(inputs: ModelInputs) -> None:
    shells = np.unique(round_shells(inputs.bvals))
    diffusion = inputs.bvals >= 50
    if inputs.bvals.size < 22 or _count_gradient_axes(inputs.bvecs[:, diffusion]) < 15:
        raise ModelInputError(
            "standard DKI needs at least 22 measurements and 15 "
            "non-collinear diffusion directions"
        )
    if shells.size < 3:
        raise ModelInputError("standard DKI needs at least three acquired shells")
    design = dki_design_matrix(
        SimpleNamespace(bvals=inputs.bvals, bvecs=inputs.bvecs.T)
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank < 22:
        raise ModelInputError(
            f"standard DKI design matrix has rank {rank}; full rank 22 is required"
        )


def _require_direct_gradients(inputs: ModelInputs) -> None:
    maximum_b = float(np.max(inputs.bvals))
    magnitude = int(np.log10(maximum_b))
    quantization = float(10 ** (magnitude - 1))
    helper_bvals = np.round(inputs.bvals / quantization) * quantization
    shells = np.unique(helper_bvals)
    design = np.column_stack(
        (
            -shells,
            shells * shells / 6.0,
            np.ones(shells.size),
        )
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank < 3 and np.unique(round_shells(inputs.bvals)).size < 3:
        raise ModelInputError(
            "Henrique direct DKI needs at least three acquired shells"
        )
    if rank < 3:
        raise ModelInputError(
            f"Henrique direct DKI shell design has rank {rank}; "
            "full rank 3 is required"
        )


def _load_avs_dki_df(path: Path) -> Callable[..., np.ndarray]:
    helper = Path(path)
    if not helper.is_file() or helper.is_symlink():
        raise ModelInputError(f"Henrique helper must be a regular file: {helper}")
    module_name = f"_dmri_henrique_{_sha256(helper)[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(helper)
    try:
        source = helper.read_bytes()
        code = compile(source, str(helper), "exec")
        exec(code, module.__dict__)
    except Exception as error:
        raise ModelInputError(f"cannot import Henrique helper: {helper}") from error
    function = getattr(module, "avs_dki_df", None)
    if not callable(function):
        raise ModelInputError("Henrique helper does not define callable avs_dki_df")
    return function


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ModelInputError(f"cannot hash Henrique helper: {path}") from error
    return digest.hexdigest()


def _require_absent(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise ModelOutputError(
            "model stage work directory contains pre-existing outputs: "
            + ", ".join(str(path.name) for path in existing)
        )


def _validate_context_identities(
    context: ModelContext, input_paths: tuple[Path, ...]
) -> None:
    lexical = [path.absolute() for path in input_paths]
    if len(set(lexical)) != len(lexical):
        raise ModelInputError("model input paths must be distinct")

    resolved = [path.resolve(strict=False) for path in input_paths]
    if len(set(resolved)) != len(resolved):
        raise ModelInputError("model inputs must not resolve to the same filesystem object")

    identities: dict[tuple[int, int], Path] = {}
    for path in input_paths:
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ModelInputError(f"cannot inspect model input path: {path}") from error
        identity = (int(stat_result.st_dev), int(stat_result.st_ino))
        if identity in identities:
            raise ModelInputError(
                "model inputs must not reference the same filesystem object: "
                f"{identities[identity]} and {path}"
            )
        identities[identity] = path

    work = context.work_dir.resolve(strict=False)
    if work in resolved:
        raise ModelInputError("model work directory must not alias an upstream input")
    for upstream in resolved:
        try:
            upstream.relative_to(work)
        except ValueError:
            continue
        raise ModelInputError(
            "model work directory must not contain an upstream input or helper"
        )
    if context.work_dir.exists() or context.work_dir.is_symlink():
        try:
            work_stat = context.work_dir.stat()
        except OSError as error:
            raise ModelInputError(
                f"cannot inspect model work directory: {context.work_dir}"
            ) from error
        work_identity = (int(work_stat.st_dev), int(work_stat.st_ino))
        if work_identity in identities:
            raise ModelInputError("model work directory must not alias an upstream input")
