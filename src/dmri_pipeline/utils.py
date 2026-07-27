"""Small, dependency-light utilities for dMRI input validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class InputAuditError(ValueError):
    """Raised when dMRI inputs do not satisfy the audit contract."""


def normalize_bvecs(array: object, n_volumes: int) -> np.ndarray:
    """Return finite b-vectors in canonical FSL 3×N orientation."""
    try:
        bvecs = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise InputAuditError("b-vectors must be numeric") from error
    if bvecs.ndim != 2:
        raise InputAuditError("b-vectors must be a 3×N or N×3 array")
    if bvecs.shape == (3, n_volumes):
        normalized = bvecs
    elif bvecs.shape == (n_volumes, 3):
        normalized = bvecs.T
    elif bvecs.shape[0] == 3 or bvecs.shape[1] == 3:
        raise InputAuditError("b-vector count must equal PA volume count")
    else:
        raise InputAuditError("b-vectors must be a 3×N or N×3 array")
    if not np.isfinite(normalized).all():
        raise InputAuditError("b-vectors must be finite")
    return normalized


def round_shells(bvals: object) -> np.ndarray:
    """Assign b0 values and round all other values to the nearest 100 s/mm²."""
    try:
        values = np.asarray(bvals, dtype=float)
    except (TypeError, ValueError) as error:
        raise InputAuditError("b-values must be numeric") from error
    if not np.isfinite(values).all():
        raise InputAuditError("b-values must be finite")
    shells = np.floor(values / 100.0 + 0.5).astype(int) * 100
    return np.where(values < 50.0, 0, shells)


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest while reading a file in fixed-size chunks."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise InputAuditError(f"Cannot hash input file: {path}") from error
    return digest.hexdigest()
