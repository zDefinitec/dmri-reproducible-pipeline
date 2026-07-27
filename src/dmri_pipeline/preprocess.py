"""Patient-generic numerical preprocessing for diffusion MRI stage workdirs."""

from __future__ import annotations

import json
import math
import os
import stat
import subprocess
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import nibabel as nib
import numpy as np
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.localpca import mppca
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError, ImageDataError
from scipy import ndimage

from .audit import InputAudit
from .config import PipelineConfig
from .utils import InputAuditError, round_shells


class PreprocessError(RuntimeError):
    """Raised when a preprocessing input, command, or output is invalid."""


class CommandResult(Protocol):
    """Minimum result contract accepted from an injected command executor."""

    returncode: int
    stderr: str | bytes | None


CommandExecutor = Callable[[Sequence[str]], CommandResult]


def _default_command_executor(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one external command without a shell and fail on a nonzero exit."""
    try:
        return subprocess.run(
            tuple(str(argument) for argument in argv),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise PreprocessError(f"BET executable was not found: {argv[0]}") from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise PreprocessError(
            f"BET failed with exit code {error.returncode}{detail}"
        ) from error
    except OSError as error:
        raise PreprocessError(f"Could not execute BET: {error}") from error


@dataclass(frozen=True)
class PreprocessContext:
    """Validated locations and dependencies for the four preprocessing stages."""

    config: PipelineConfig
    audit: InputAudit
    denoise_dir: Path
    gibbs_dir: Path
    topup_dir: Path
    bet_dir: Path
    bet_mask_source: Path
    command_executor: CommandExecutor = _default_command_executor

    def __post_init__(self) -> None:
        if not isinstance(self.config, PipelineConfig):
            raise PreprocessError("context config must be a PipelineConfig")
        if not isinstance(self.audit, InputAudit):
            raise PreprocessError("context audit must be an InputAudit")
        locations = (
            self.denoise_dir,
            self.gibbs_dir,
            self.topup_dir,
            self.bet_dir,
            self.bet_mask_source,
        )
        if any(not isinstance(location, Path) for location in locations):
            raise PreprocessError("preprocessing locations must be pathlib.Path values")
        _validate_context_paths(self)
        if not callable(self.command_executor):
            raise PreprocessError("command_executor must be callable")
        acquisition = self.config.acquisition
        readout = acquisition.total_readout_time
        if isinstance(readout, bool) or not isinstance(readout, Real):
            raise PreprocessError("total readout time must be a real number")
        if (
            not math.isfinite(readout)
            or readout <= 0
        ):
            raise PreprocessError("total readout time must be positive and finite")
        if isinstance(acquisition.slice_axis, bool) or not isinstance(
            acquisition.slice_axis, int
        ):
            raise PreprocessError("slice axis must be an integer: 0, 1, or 2")
        if acquisition.slice_axis not in (0, 1, 2):
            raise PreprocessError("slice axis must be 0, 1, or 2")
        for label, vector in (
            ("PA", acquisition.pa_vector),
            ("AP", acquisition.ap_vector),
        ):
            if (
                len(vector) != 3
                or any(
                    isinstance(component, bool) or not isinstance(component, int)
                    for component in vector
                )
                or sum(abs(component) for component in vector) != 1
            ):
                raise PreprocessError(
                    f"{label} phase-encoding vector must be a signed cardinal axis"
                )
        if acquisition.pa_vector == acquisition.ap_vector:
            raise PreprocessError("PA and AP phase-encoding vectors must be distinct")

    @property
    def cpu_count(self) -> int:
        """Return a positive process ceiling derived from the current machine."""
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class DenoiseDetails:
    denoised_pa: Path
    denoised_ap: Path
    sigma_pa: Path
    sigma_ap: Path
    raw_mean_b0: Path
    raw_bet_mask: Path
    dilated_mask: Path
    metrics: Path
    pa_patch_radius: int


@dataclass(frozen=True)
class GibbsDetails:
    corrected_pa: Path
    corrected_ap: Path
    metrics: Path
    process_count: int


@dataclass(frozen=True)
class TopupInputDetails:
    pa_b0: Path
    ap_b0: Path
    merged_b0: Path
    acqparams_topup: Path
    acqparams_eddy: Path
    index_eddy: Path
    bvals_rounded: Path
    manifest: Path
    pa_b0_count: int
    ap_b0_count: int


@dataclass(frozen=True)
class MaskCleanupDetails:
    cleaned_mask: Path
    metrics: Path
    component_count: int
    original_voxel_count: int
    largest_voxel_count: int
    removed_voxel_count: int
    largest_tie: bool


def choose_patch_radius(n_measurements: int) -> int:
    """Return the smallest radius whose odd cubic patch exceeds the samples."""
    if (
        isinstance(n_measurements, bool)
        or not isinstance(n_measurements, int)
        or n_measurements <= 0
    ):
        raise PreprocessError("n_measurements must be a positive integer")
    radius = 0
    while (2 * radius + 1) ** 3 <= n_measurements:
        radius += 1
    return radius


def run_denoise(context: PreprocessContext) -> DenoiseDetails:
    """Run masked PA and full-grid AP MP-PCA inside the denoise workdir."""
    _require_context(context)
    output_dir = context.denoise_dir
    paths = {
        "denoised_pa": output_dir / "denoised_PA.nii.gz",
        "denoised_ap": output_dir / "denoised_AP.nii.gz",
        "sigma_pa": output_dir / "sigma_PA.nii.gz",
        "sigma_ap": output_dir / "sigma_AP.nii.gz",
        "raw_mean_b0": output_dir / "raw_mean_b0.nii.gz",
        "bet_brain": output_dir / "raw_mean_b0_bet.nii.gz",
        "raw_bet_mask": output_dir / "raw_mean_b0_bet_mask.nii.gz",
        "dilated_mask": output_dir / "raw_mean_b0_bet_mask_dilated.nii.gz",
        "metrics": output_dir / "denoise_metrics.json",
    }
    _ensure_outputs_absent(paths.values())
    output_dir.mkdir(parents=True, exist_ok=True)

    pa_image, pa = _load_float_image(context.config.dwi_pa, "raw PA image")
    if pa.ndim != 4:
        raise PreprocessError("raw PA image must be finite 4D data")
    bvals = _load_bvals(context.config.bvals)
    if bvals.size != pa.shape[3]:
        raise PreprocessError("b-value count must equal PA volume count")
    b0_mask = bvals < 50.0
    if not b0_mask.any():
        raise PreprocessError("PA data must contain at least one b0 with b < 50")
    _validate_against_audit(context, pa_image, pa.shape, "PA")

    ap_image, ap = _load_float_image(context.config.b0_ap, "raw AP image")
    if ap.ndim == 3:
        ap = ap[..., None]
    elif ap.ndim != 4:
        raise PreprocessError("raw AP image must be finite 3D or 4D data")
    if ap.shape[:3] != pa.shape[:3] or not np.allclose(
        ap_image.affine, pa_image.affine, atol=1e-5, rtol=0.0
    ):
        raise PreprocessError("raw PA and AP images must use the same image grid")
    _validate_against_audit(context, ap_image, ap_image.shape, "AP")

    raw_mean = np.mean(pa[..., b0_mask], axis=-1, dtype=np.float32)
    _save_image(
        raw_mean,
        pa_image,
        paths["raw_mean_b0"],
        dtype=np.float32,
        description="Raw PA mean b0 for early BET",
    )
    bet_executable = (
        str(context.config.fsldir / "bin" / "bet")
        if context.config.fsldir is not None
        else "bet"
    )
    bet_prefix = paths["bet_brain"].with_suffix("").with_suffix("")
    argv = (
        bet_executable,
        str(paths["raw_mean_b0"]),
        str(bet_prefix),
        "-R",
        "-f",
        "0.25",
        "-g",
        "0",
        "-m",
    )
    try:
        result = context.command_executor(argv)
    except PreprocessError:
        raise
    except Exception as error:
        raise PreprocessError(f"BET execution failed: {error}") from error
    returncode = getattr(result, "returncode", None)
    if returncode != 0:
        stderr_value = getattr(result, "stderr", "")
        if isinstance(stderr_value, bytes):
            stderr = stderr_value.decode(errors="replace").strip()
        else:
            stderr = str(stderr_value or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise PreprocessError(f"BET failed with exit code {returncode}{detail}")

    mask_image, raw_mask_data = _load_float_image(
        paths["raw_bet_mask"], "early BET mask"
    )
    if raw_mask_data.ndim != 3:
        raise PreprocessError("early BET mask must be finite 3D data")
    if raw_mask_data.shape != pa.shape[:3] or not np.allclose(
        mask_image.affine, pa_image.affine, atol=1e-5, rtol=0.0
    ):
        raise PreprocessError("early BET mask and PA image must use the same 3D grid")
    raw_foreground = raw_mask_data > 0
    if not raw_foreground.any():
        raise PreprocessError("early BET mask must contain foreground voxels")
    _save_image(
        raw_foreground,
        mask_image,
        paths["raw_bet_mask"],
        dtype=np.uint8,
        description="Raw early BET brain mask",
    )
    computation_mask = ndimage.binary_dilation(
        raw_foreground,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=2,
    )
    _save_image(
        computation_mask,
        pa_image,
        paths["dilated_mask"],
        dtype=np.uint8,
        description="Early BET mask dilated by two 26-connected iterations",
    )

    pa_radius = choose_patch_radius(int(pa.shape[3]))
    original_pa_outside_mask = np.array(pa[~computation_mask], copy=True)
    pa_denoised, pa_sigma = mppca(
        pa,
        mask=computation_mask,
        patch_radius=pa_radius,
        pca_method="eig",
        return_sigma=True,
        out_dtype=np.float32,
        suppress_warning=False,
    )
    ap_denoised, ap_sigma = mppca(
        ap,
        mask=None,
        patch_radius=2,
        pca_method="eig",
        return_sigma=True,
        out_dtype=np.float32,
        suppress_warning=False,
    )
    pa_denoised = _finite_array(pa_denoised, "PA MP-PCA output")
    pa_sigma = _finite_array(pa_sigma, "PA MP-PCA sigma")
    ap_denoised = _finite_array(ap_denoised, "AP MP-PCA output")
    ap_sigma = _finite_array(ap_sigma, "AP MP-PCA sigma")
    if pa_denoised.shape != pa.shape or ap_denoised.shape != ap.shape:
        raise PreprocessError("MP-PCA output shape must match its input")
    if pa_sigma.shape != pa.shape[:3] or ap_sigma.shape != ap.shape[:3]:
        raise PreprocessError("MP-PCA sigma output must match the spatial grid")
    pa_denoised = np.asarray(pa_denoised, dtype=np.float32)
    pa_denoised[~computation_mask] = original_pa_outside_mask

    _save_image(
        pa_denoised,
        pa_image,
        paths["denoised_pa"],
        dtype=np.float32,
        description=f"PA DIPY MP-PCA denoised; patch radius {pa_radius}",
    )
    _save_image(
        ap_denoised,
        ap_image,
        paths["denoised_ap"],
        dtype=np.float32,
        description="AP DIPY MP-PCA denoised; patch radius 2",
    )
    _save_image(
        pa_sigma,
        pa_image,
        paths["sigma_pa"],
        dtype=np.float32,
        description="PA MP-PCA spatial noise sigma",
    )
    _save_image(
        ap_sigma,
        ap_image,
        paths["sigma_ap"],
        dtype=np.float32,
        description="AP MP-PCA spatial noise sigma",
    )
    _write_json(
        paths["metrics"],
        {
            "ap_measurement_count": int(ap.shape[3]),
            "ap_patch_radius": 2,
            "dilation_connectivity": 26,
            "dilation_iterations": 2,
            "pa_b0_count": int(np.count_nonzero(b0_mask)),
            "pa_computation_mask_voxels": int(computation_mask.sum()),
            "pa_measurement_count": int(pa.shape[3]),
            "pa_patch_radius": pa_radius,
        },
    )
    return DenoiseDetails(
        denoised_pa=paths["denoised_pa"],
        denoised_ap=paths["denoised_ap"],
        sigma_pa=paths["sigma_pa"],
        sigma_ap=paths["sigma_ap"],
        raw_mean_b0=paths["raw_mean_b0"],
        raw_bet_mask=paths["raw_bet_mask"],
        dilated_mask=paths["dilated_mask"],
        metrics=paths["metrics"],
        pa_patch_radius=pa_radius,
    )


def run_gibbs(context: PreprocessContext) -> GibbsDetails:
    """Apply Gibbs removal to denoised PA/AP without mutating stage inputs."""
    _require_context(context)
    source_pa = context.denoise_dir / "denoised_PA.nii.gz"
    source_ap = context.denoise_dir / "denoised_AP.nii.gz"
    output_dir = context.gibbs_dir
    corrected_pa = output_dir / "gibbs_PA.nii.gz"
    corrected_ap = output_dir / "gibbs_AP.nii.gz"
    metrics_path = output_dir / "gibbs_metrics.json"
    _ensure_outputs_absent((corrected_pa, corrected_ap, metrics_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    pa_image, pa_loaded = _load_float_image(source_pa, "denoised PA image")
    ap_image, ap_loaded = _load_float_image(source_ap, "denoised AP image")
    if pa_loaded.ndim != 4 or ap_loaded.ndim != 4:
        raise PreprocessError("denoised PA and AP inputs must be finite 4D data")
    if pa_loaded.shape != context.audit.pa_shape:
        raise PreprocessError("denoised PA shape does not match the input audit")
    expected_ap_shape = (*context.audit.ap_shape[:3], context.audit.ap_b0_count)
    if ap_loaded.shape != expected_ap_shape:
        raise PreprocessError("denoised AP shape does not match the input audit")
    if pa_loaded.shape[:3] != ap_loaded.shape[:3] or not np.allclose(
        pa_image.affine, ap_image.affine, atol=1e-5, rtol=0.0
    ):
        raise PreprocessError("denoised PA and AP images must use the same image grid")

    pa = np.array(pa_loaded, dtype=np.float32, copy=True)
    ap = np.array(ap_loaded, dtype=np.float32, copy=True)
    process_count = min(12, context.cpu_count)
    corrected_pa_data = gibbs_removal(
        pa,
        slice_axis=context.config.acquisition.slice_axis,
        n_points=3,
        inplace=True,
        num_processes=process_count,
    )
    corrected_ap_data = gibbs_removal(
        ap,
        slice_axis=context.config.acquisition.slice_axis,
        n_points=3,
        inplace=True,
        num_processes=process_count,
    )
    corrected_pa_array = _finite_array(corrected_pa_data, "PA Gibbs output")
    corrected_ap_array = _finite_array(corrected_ap_data, "AP Gibbs output")
    if corrected_pa_array.shape != pa_loaded.shape or corrected_ap_array.shape != ap_loaded.shape:
        raise PreprocessError("Gibbs output shape must match its denoised input")

    _save_image(
        corrected_pa_array,
        pa_image,
        corrected_pa,
        dtype=np.float32,
        description="PA DIPY Gibbs ringing correction; n_points 3",
    )
    _save_image(
        corrected_ap_array,
        ap_image,
        corrected_ap,
        dtype=np.float32,
        description="AP DIPY Gibbs ringing correction; n_points 3",
    )
    _write_json(
        metrics_path,
        {
            "n_points": 3,
            "process_count": process_count,
            "slice_axis": context.config.acquisition.slice_axis,
        },
    )
    return GibbsDetails(corrected_pa, corrected_ap, metrics_path, process_count)


def prepare_topup_inputs(context: PreprocessContext) -> TopupInputDetails:
    """Write ordered PA/AP b0 images and exact TOPUP/EDDY parameter files."""
    _require_context(context)
    source_pa = context.gibbs_dir / "gibbs_PA.nii.gz"
    source_ap = context.gibbs_dir / "gibbs_AP.nii.gz"
    output_dir = context.topup_dir
    paths = {
        "pa_b0": output_dir / "nodif_PA_all.nii.gz",
        "ap_b0": output_dir / "nodif_AP_all.nii.gz",
        "merged_b0": output_dir / "PA_AP_b0.nii.gz",
        "acqparams_topup": output_dir / "acqparams_topup.txt",
        "acqparams_eddy": output_dir / "acqparams_eddy.txt",
        "index_eddy": output_dir / "index_eddy.txt",
        "bvals_rounded": output_dir / "bvals_rounded",
        "manifest": output_dir / "topup_input_manifest.json",
    }
    _ensure_outputs_absent(paths.values())
    output_dir.mkdir(parents=True, exist_ok=True)

    pa_image, pa = _load_float_image(source_pa, "Gibbs-corrected PA image")
    ap_image, ap = _load_float_image(source_ap, "Gibbs-corrected AP image")
    if pa.ndim != 4 or ap.ndim != 4:
        raise PreprocessError("Gibbs-corrected PA and AP images must be finite 4D data")
    expected_ap_shape = (*context.audit.ap_shape[:3], context.audit.ap_b0_count)
    if pa.shape != context.audit.pa_shape or ap.shape != expected_ap_shape:
        raise PreprocessError(
            "Gibbs-corrected PA/AP shapes must match the input audit"
        )
    if not np.allclose(
        pa_image.affine,
        np.asarray(context.audit.pa_affine),
        atol=1e-5,
        rtol=0.0,
    ) or not np.allclose(
        ap_image.affine,
        np.asarray(context.audit.ap_affine),
        atol=1e-5,
        rtol=0.0,
    ):
        raise PreprocessError(
            "Gibbs-corrected PA/AP affines must match the input audit"
        )
    if pa.shape[:3] != ap.shape[:3] or not np.allclose(
        pa_image.affine, ap_image.affine, atol=1e-5, rtol=0.0
    ):
        raise PreprocessError(
            "Gibbs-corrected PA and AP images must use the same image grid"
        )
    bvals = _load_bvals(context.config.bvals)
    if bvals.size != pa.shape[3]:
        raise PreprocessError("b-value count must equal corrected PA volume count")
    pa_b0_indices = np.flatnonzero(bvals < 50.0)
    if pa_b0_indices.size == 0:
        raise PreprocessError("corrected PA data must contain at least one b0")
    if ap.shape[3] == 0:
        raise PreprocessError("corrected AP data must contain at least one b0")

    pa_b0 = np.asarray(pa[..., pa_b0_indices], dtype=np.float32)
    ap_b0 = np.asarray(ap, dtype=np.float32)
    merged = np.concatenate((pa_b0, ap_b0), axis=3)
    _save_image(
        pa_b0,
        pa_image,
        paths["pa_b0"],
        dtype=np.float32,
        description="All corrected PA b0 volumes",
    )
    _save_image(
        ap_b0,
        ap_image,
        paths["ap_b0"],
        dtype=np.float32,
        description="All corrected AP b0 volumes",
    )
    _save_image(
        merged,
        pa_image,
        paths["merged_b0"],
        dtype=np.float32,
        description="Corrected PA then AP b0 volumes for TOPUP",
    )

    pa_row = _acquisition_row(
        context.config.acquisition.pa_vector,
        context.config.acquisition.total_readout_time,
    )
    ap_row = _acquisition_row(
        context.config.acquisition.ap_vector,
        context.config.acquisition.total_readout_time,
    )
    topup_rows = [pa_row] * int(pa_b0.shape[3]) + [ap_row] * int(ap_b0.shape[3])
    _write_text(paths["acqparams_topup"], "\n".join(topup_rows) + "\n")
    _write_text(paths["acqparams_eddy"], f"{pa_row}\n{ap_row}\n")
    _write_text(paths["index_eddy"], " ".join(["1"] * int(pa.shape[3])) + "\n")
    try:
        rounded_bvals = round_shells(bvals)
    except InputAuditError as error:
        raise PreprocessError(f"Cannot round b-values: {error}") from error
    _write_text(
        paths["bvals_rounded"],
        " ".join(str(int(value)) for value in rounded_bvals) + "\n",
    )
    _write_json(
        paths["manifest"],
        {
            "ap_acquisition_row": [
                *context.config.acquisition.ap_vector,
                context.config.acquisition.total_readout_time,
            ],
            "ap_b0_count": int(ap_b0.shape[3]),
            "combined_b0_count": int(merged.shape[3]),
            "eddy_acquisition_row_order": ["PA", "AP"],
            "eddy_index_count": int(pa.shape[3]),
            "pa_acquisition_row": [
                *context.config.acquisition.pa_vector,
                context.config.acquisition.total_readout_time,
            ],
            "pa_b0_count": int(pa_b0.shape[3]),
            "pa_b0_indices": [int(index) for index in pa_b0_indices],
            "volume_order": ["PA"] * int(pa_b0.shape[3])
            + ["AP"] * int(ap_b0.shape[3]),
        },
    )
    return TopupInputDetails(
        pa_b0=paths["pa_b0"],
        ap_b0=paths["ap_b0"],
        merged_b0=paths["merged_b0"],
        acqparams_topup=paths["acqparams_topup"],
        acqparams_eddy=paths["acqparams_eddy"],
        index_eddy=paths["index_eddy"],
        bvals_rounded=paths["bvals_rounded"],
        manifest=paths["manifest"],
        pa_b0_count=int(pa_b0.shape[3]),
        ap_b0_count=int(ap_b0.shape[3]),
    )


def clean_bet_mask(context: PreprocessContext) -> MaskCleanupDetails:
    """Retain the deterministic largest 26-connected BET mask component."""
    _require_context(context)
    cleaned_path = context.bet_dir / "nodif_brain_mask.nii.gz"
    metrics_path = context.bet_dir / "mask_cleanup_metrics.json"
    if _lexical_absolute(context.bet_mask_source) == _lexical_absolute(cleaned_path):
        raise PreprocessError("raw BET mask source must differ from cleaned output")
    _ensure_outputs_absent((cleaned_path, metrics_path))
    context.bet_dir.mkdir(parents=True, exist_ok=True)

    image, data = _load_float_image(context.bet_mask_source, "raw BET mask")
    if data.ndim != 3:
        raise PreprocessError("raw BET mask must be finite 3D data")
    foreground = data > 0
    original_voxels = int(foreground.sum())
    if original_voxels == 0:
        raise PreprocessError("raw BET mask must contain at least one foreground voxel")
    labels, component_count = ndimage.label(
        foreground,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel(), minlength=int(component_count) + 1)[1:]
    largest_size = int(sizes.max())
    largest_labels = np.flatnonzero(sizes == largest_size) + 1
    selected_component = int(largest_labels[0])
    largest_tie = bool(largest_labels.size > 1)
    cleaned = labels == selected_component
    removed_voxels = original_voxels - largest_size

    _save_image(
        cleaned,
        image,
        cleaned_path,
        dtype=np.uint8,
        description="Largest 26-connected BET mask component",
    )
    metrics = {
        "component_count": int(component_count),
        "largest_voxel_count": largest_size,
        "largest_tie": largest_tie,
        "original_voxel_count": original_voxels,
        "removed_voxel_count": removed_voxels,
        "selected_component": selected_component,
    }
    _write_json(metrics_path, metrics)
    return MaskCleanupDetails(
        cleaned_mask=cleaned_path,
        metrics=metrics_path,
        component_count=int(component_count),
        original_voxel_count=original_voxels,
        largest_voxel_count=largest_size,
        removed_voxel_count=removed_voxels,
        largest_tie=largest_tie,
    )


def _require_context(context: object) -> PreprocessContext:
    if not isinstance(context, PreprocessContext):
        raise PreprocessError("context must be a PreprocessContext")
    _validate_context_paths(context)
    return context


def _load_float_image(
    path: Path, label: str
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    try:
        image = nib.load(path)
        data = np.asarray(image.dataobj, dtype=np.float32)
    except (OSError, ImageFileError, HeaderDataError, ImageDataError, ValueError) as error:
        raise PreprocessError(f"Cannot read {label}: {path}") from error
    if not np.isfinite(image.affine).all() or not np.isfinite(data).all():
        raise PreprocessError(f"{label} and its affine must contain only finite values")
    return image, data


def _load_bvals(path: Path) -> np.ndarray:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise PreprocessError(f"Cannot read b-values: {path}") from error
    tokens = text.split()
    if not tokens:
        raise PreprocessError("b-values file must not be empty")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError as error:
        raise PreprocessError("b-values must contain only numeric values") from error
    if not np.isfinite(values).all():
        raise PreprocessError("b-values must contain only finite values")
    return values


def _validate_against_audit(
    context: PreprocessContext,
    image: nib.spatialimages.SpatialImage,
    shape: tuple[int, ...],
    label: str,
) -> None:
    expected_shape = context.audit.pa_shape if label == "PA" else context.audit.ap_shape
    expected_affine = context.audit.pa_affine if label == "PA" else context.audit.ap_affine
    if tuple(int(size) for size in shape) != expected_shape:
        raise PreprocessError(f"{label} image shape does not match the input audit")
    if not np.allclose(
        image.affine, np.asarray(expected_affine), atol=1e-5, rtol=0.0
    ):
        raise PreprocessError(f"{label} image affine does not match the input audit")


def _finite_array(array: object, label: str) -> np.ndarray:
    try:
        values = np.asarray(array)
    except (TypeError, ValueError) as error:
        raise PreprocessError(f"{label} must be a numeric array") from error
    if not np.isfinite(values).all():
        raise PreprocessError(f"{label} must contain only finite values")
    return values


def _ensure_outputs_absent(paths: Iterable[Path]) -> None:
    for path in paths:
        destination = Path(path)
        _validate_no_symlink_components(
            destination.parent,
            f"output parent for {destination.name}",
            final_must_be_directory=True,
        )
        if os.path.lexists(destination):
            raise PreprocessError(
                "Refusing to overwrite destination that already exists: "
                f"{destination}"
            )


def _validate_context_paths(context: PreprocessContext) -> None:
    stage_paths = (
        ("denoise stage directory", context.denoise_dir),
        ("Gibbs stage directory", context.gibbs_dir),
        ("TOPUP stage directory", context.topup_dir),
        ("BET stage directory", context.bet_dir),
    )
    lexical_stage_paths: list[Path] = []
    existing_identities: dict[tuple[int, int], str] = {}
    for label, path in stage_paths:
        lexical = _validate_no_symlink_components(
            path,
            label,
            final_must_be_directory=True,
        )
        lexical_stage_paths.append(lexical)
        try:
            metadata = os.lstat(lexical)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PreprocessError(f"Cannot inspect {label}: {path}") from error
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in existing_identities:
            raise PreprocessError(
                "preprocessing stage directories must be distinct and must not "
                f"alias each other: {existing_identities[identity]} and {label}"
            )
        existing_identities[identity] = label
    if len(set(lexical_stage_paths)) != len(lexical_stage_paths):
        raise PreprocessError(
            "preprocessing stage directories must be distinct and must not alias "
            "each other"
        )
    _validate_no_symlink_components(
        context.bet_mask_source,
        "raw BET mask source",
        final_must_be_directory=False,
    )


def _lexical_absolute(path: Path) -> Path:
    if os.pardir in path.parts:
        raise PreprocessError(
            f"filesystem path must not contain parent traversal ('..'): {path}"
        )
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError) as error:
        raise PreprocessError(f"Invalid filesystem path: {path!r}") from error


def _validate_no_symlink_components(
    path: Path,
    label: str,
    *,
    final_must_be_directory: bool,
) -> Path:
    lexical = _lexical_absolute(path)
    current = Path(lexical.anchor)
    parts = lexical.parts[1:]
    for index, component in enumerate(parts):
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PreprocessError(f"Cannot inspect {label}: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise PreprocessError(
                f"{label} must not contain a symbolic link component: {current}"
            )
        is_final = index == len(parts) - 1
        if (not is_final or final_must_be_directory) and not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise PreprocessError(
                f"{label} requires directory components, but {current} is not "
                "a directory"
            )
    return lexical


def _save_image(
    data: object,
    reference: nib.spatialimages.SpatialImage,
    destination: Path,
    *,
    dtype: np.dtype | type[np.generic],
    description: str,
) -> None:
    values = _finite_array(data, f"output {destination.name}")
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header["descrip"] = description[:79].encode("ascii", errors="ignore")
    try:
        nib.save(
            nib.Nifti1Image(np.asarray(values, dtype=dtype), reference.affine, header),
            destination,
        )
    except (OSError, HeaderDataError, ImageDataError, ValueError) as error:
        raise PreprocessError(f"Cannot write NIfTI output: {destination}") from error


def _write_json(destination: Path, payload: dict[str, object]) -> None:
    _write_text(
        destination,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_text(destination: Path, content: str) -> None:
    try:
        destination.write_text(content, encoding="utf-8")
    except OSError as error:
        raise PreprocessError(f"Cannot write output: {destination}") from error


def _acquisition_row(vector: tuple[int, int, int], readout: float) -> str:
    return " ".join((*[str(component) for component in vector], repr(float(readout))))
