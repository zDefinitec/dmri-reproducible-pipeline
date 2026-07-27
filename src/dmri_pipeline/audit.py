"""Read-only validation and provenance reporting for diffusion MRI inputs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError, ImageDataError

from .config import PipelineConfig
from .utils import InputAuditError, normalize_bvecs, round_shells, sha256_file


@dataclass(frozen=True)
class InputAudit:
    """Immutable, path-free summary of validated raw dMRI inputs."""

    pa_shape: tuple[int, ...]
    ap_shape: tuple[int, ...]
    ap_b0_count: int
    shell_counts: Mapping[int, int]
    hashes: Mapping[str, str]
    pa_affine: tuple[tuple[float, ...], ...]
    ap_affine: tuple[tuple[float, ...], ...]
    pa_spatial_zooms: tuple[float, float, float]
    ap_spatial_zooms: tuple[float, float, float]
    b0_indices: tuple[int, ...]
    gradient_norm_range: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable representation."""
        return {
            "ap_affine": [list(row) for row in self.ap_affine],
            "ap_b0_count": self.ap_b0_count,
            "ap_shape": list(self.ap_shape),
            "ap_spatial_zooms": list(self.ap_spatial_zooms),
            "b0_indices": list(self.b0_indices),
            "gradient_norm_range": list(self.gradient_norm_range),
            "hashes": dict(sorted(self.hashes.items())),
            "pa_affine": [list(row) for row in self.pa_affine],
            "pa_shape": list(self.pa_shape),
            "pa_spatial_zooms": list(self.pa_spatial_zooms),
            "shell_counts": {
                str(shell): count for shell, count in sorted(self.shell_counts.items())
            },
        }


def audit_inputs(config: PipelineConfig) -> InputAudit:
    """Validate raw dMRI input compatibility and return path-free provenance."""
    _validate_acquisition(config)
    pa = _load_image(config.dwi_pa, "PA")
    ap = _load_image(config.b0_ap, "AP")
    if len(pa.shape) != 4:
        raise InputAuditError("PA image must be 4D")
    if len(ap.shape) not in (3, 4):
        raise InputAuditError("AP image must be 3D or 4D")
    if pa.shape[:3] != ap.shape[:3]:
        raise InputAuditError("PA and AP images must have the same spatial shape")
    if not np.allclose(pa.affine, ap.affine, atol=1e-5, rtol=0.0):
        raise InputAuditError("PA and AP images must use the same image grid")

    n_volumes = pa.shape[3]
    bvals = _load_bvals(config.bvals)
    if bvals.size != n_volumes:
        raise InputAuditError("b-value count must equal PA volume count")
    bvecs = normalize_bvecs(_load_text(config.bvecs, "b-vectors"), n_volumes)
    b0_mask = bvals < 50.0
    if not b0_mask.any():
        raise InputAuditError("Inputs must contain at least one b0 with b < 50")

    norms = np.linalg.norm(bvecs, axis=0)
    accepted_b0 = (norms[b0_mask] < 0.1) | (
        (norms[b0_mask] >= 0.95) & (norms[b0_mask] <= 1.05)
    )
    if not accepted_b0.all():
        raise InputAuditError("b0 vectors must be near-zero or unit length")
    diffusion_mask = ~b0_mask
    if np.any((norms[diffusion_mask] < 0.95) | (norms[diffusion_mask] > 1.05)):
        raise InputAuditError("non-b0 vectors must have unit length (norm 0.95 to 1.05)")
    if _count_gradient_axes(bvecs[:, diffusion_mask]) < 6:
        raise InputAuditError("Need at least six unique non-collinear non-b0 gradient axes")

    shells = round_shells(bvals)
    shell_counts = {
        int(shell): int(np.count_nonzero(shells == shell)) for shell in np.unique(shells)
    }
    return InputAudit(
        pa_shape=tuple(int(size) for size in pa.shape),
        ap_shape=tuple(int(size) for size in ap.shape),
        ap_b0_count=1 if len(ap.shape) == 3 else int(ap.shape[3]),
        shell_counts=MappingProxyType(shell_counts),
        hashes=MappingProxyType(
            {
                "ap": sha256_file(config.b0_ap),
                "bvals": sha256_file(config.bvals),
                "bvecs": sha256_file(config.bvecs),
                "pa": sha256_file(config.dwi_pa),
            }
        ),
        pa_affine=_affine_tuple(pa.affine),
        ap_affine=_affine_tuple(ap.affine),
        pa_spatial_zooms=_spatial_zooms(pa),
        ap_spatial_zooms=_spatial_zooms(ap),
        b0_indices=tuple(int(index) for index in np.flatnonzero(b0_mask)),
        gradient_norm_range=(float(norms.min()), float(norms.max())),
    )


def write_input_audit(audit: InputAudit, path: Path) -> None:
    """Atomically replace *path* with a deterministic audit JSON document."""
    destination = Path(path)
    payload = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
        os.replace(temporary_name, destination)
    except OSError as error:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise InputAuditError(f"Cannot write input audit: {destination}") from error


def _load_image(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(path)
    except (OSError, ImageFileError, HeaderDataError, ImageDataError) as error:
        raise InputAuditError(f"Cannot read {label} image") from error
    if not np.isfinite(image.affine).all():
        raise InputAuditError(f"{label} image affine must be finite")
    return image


def _load_text(path: Path, label: str) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=float)
    except (OSError, ValueError) as error:
        raise InputAuditError(f"Cannot read {label}") from error
    return np.asarray(values, dtype=float)


def _load_bvals(path: Path) -> np.ndarray:
    bvals = _load_text(path, "b-values").reshape(-1)
    if not np.isfinite(bvals).all():
        raise InputAuditError("b-values must be finite")
    return bvals


def _validate_acquisition(config: PipelineConfig) -> None:
    try:
        values = np.asarray(
            (*config.acquisition.pa_vector, *config.acquisition.ap_vector,
             config.acquisition.total_readout_time, config.acquisition.slice_axis),
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise InputAuditError("configuration acquisition values must be finite") from error
    if not np.isfinite(values).all():
        raise InputAuditError("configuration acquisition values must be finite")


def _count_gradient_axes(vectors: np.ndarray) -> int:
    axes: list[np.ndarray] = []
    for vector in vectors.T:
        axis = vector / np.linalg.norm(vector)
        first_nonzero = np.flatnonzero(np.abs(axis) > 1e-8)[0]
        if axis[first_nonzero] < 0:
            axis = -axis
        if not any(np.allclose(axis, existing, atol=1e-5, rtol=0.0) for existing in axes):
            axes.append(axis)
    return len(axes)


def _affine_tuple(affine: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in affine)


def _spatial_zooms(image: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    zooms = image.header.get_zooms()[:3]
    if not np.isfinite(zooms).all():
        raise InputAuditError("image spatial zooms must be finite")
    return tuple(float(value) for value in zooms)  # type: ignore[return-value]
