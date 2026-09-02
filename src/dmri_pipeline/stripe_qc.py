"""Patient-generic pre-denoise motion/stripe quality control.

``compute_stripe_indices`` accepts either a NiBabel spatial image or a 4D
NumPy array. NiBabel inputs are read one 3D volume at a time through their data
proxy so the complete raw DWI is not duplicated in memory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.signal import convolve2d

from .config import PipelineConfig
from .utils import round_shells


_STRIPE_KERNEL_BYTES = np.tile(
    [0.0, -1.0, 2.0, -1.0, 0.0], (5, 1)
).tobytes()
STRIPE_KERNEL = np.frombuffer(_STRIPE_KERNEL_BYTES, dtype=np.float64).reshape(5, 5)

_AMBIGUOUS_THRESHOLD = 1.15
_HIGH_THRESHOLD = 1.25
_EXCLUDE_HIGH_COUNT = 5
_METHOD_NAME = "Henrique Appendix A sagittal stripe index"
_STATUSES = {
    "INCLUDE",
    "INCLUDE_WITH_FLAGS",
    "INCLUDE_AFTER_REVIEW",
    "HOLD_FOR_REVIEW",
    "EXCLUDE",
}


class StripeQCError(ValueError):
    """Raised when stripe QC cannot produce a scientifically valid result."""


@dataclass(frozen=True)
class StripeMetrics:
    """Immutable per-volume aSI/cSI measurements and their locations."""

    a_si: np.ndarray
    c_si: np.ndarray
    shells: np.ndarray
    peak_sagittal: np.ndarray

    def __post_init__(self) -> None:
        a_si = _readonly_float_array(self.a_si, "a_si")
        c_si = _readonly_float_array(self.c_si, "c_si")
        shells = _readonly_integer_array(self.shells, "shells")
        peak = _readonly_integer_array(self.peak_sagittal, "peak_sagittal")
        lengths = {array.size for array in (a_si, c_si, shells, peak)}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise StripeQCError("Stripe metric arrays must have the same nonzero length")
        if np.any(a_si < 0):
            raise StripeQCError("a_si values must be nonnegative")
        if np.any(c_si <= 0):
            raise StripeQCError("c_si values must be positive")
        if np.any(shells < 0):
            raise StripeQCError("shell values must be nonnegative")
        if np.any(peak < 0):
            raise StripeQCError("peak_sagittal indices must be nonnegative")
        object.__setattr__(self, "a_si", a_si)
        object.__setattr__(self, "c_si", c_si)
        object.__setattr__(self, "shells", shells)
        object.__setattr__(self, "peak_sagittal", peak)


@dataclass(frozen=True)
class QCDecision:
    """Immutable gate result using zero-based indices."""

    status: Literal[
        "INCLUDE",
        "INCLUDE_WITH_FLAGS",
        "INCLUDE_AFTER_REVIEW",
        "HOLD_FOR_REVIEW",
        "EXCLUDE",
    ]
    high_indices: tuple[int, ...]
    ambiguous_indices: tuple[int, ...]
    exit_code: int

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise StripeQCError(f"Unknown QC decision status: {self.status}")
        high = _validated_indices(self.high_indices, "high_indices")
        ambiguous = _validated_indices(self.ambiguous_indices, "ambiguous_indices")
        if set(high) & set(ambiguous):
            raise StripeQCError("High and ambiguous indices must not overlap")
        expected_exit = 20 if self.status == "EXCLUDE" else 21 if self.status == "HOLD_FOR_REVIEW" else 0
        if isinstance(self.exit_code, bool) or self.exit_code != expected_exit:
            raise StripeQCError(
                f"{self.status} requires exit code {expected_exit}, got {self.exit_code}"
            )
        object.__setattr__(self, "high_indices", high)
        object.__setattr__(self, "ambiguous_indices", ambiguous)


def compute_stripe_indices(image: object, bvals: object) -> StripeMetrics:
    """Compute aSI and within-shell cSI for a raw 4D PA DWI.

    Each sagittal slice is convolved independently with ``STRIPE_KERNEL`` in
    full mode, matching MATLAB ``conv2`` and SciPy ``convolve2d`` semantics.
    """

    shape, data = _image_shape_and_data(image)
    if len(shape) != 4:
        raise StripeQCError(f"Stripe QC requires a 4D PA DWI; found shape {shape}")
    if any(int(length) <= 0 for length in shape):
        raise StripeQCError(f"Stripe QC cannot process empty image dimensions: {shape}")

    values = _validated_bvals(bvals, int(shape[3]))
    a_si = np.empty(shape[3], dtype=np.float64)
    peak_sagittal = np.empty(shape[3], dtype=np.int64)

    for volume_index in range(shape[3]):
        try:
            volume = np.asarray(data[..., volume_index], dtype=np.float64)
        except (IndexError, TypeError, ValueError, OSError) as error:
            raise StripeQCError(
                f"Cannot read raw PA DWI volume {volume_index + 1}"
            ) from error
        if volume.shape != tuple(shape[:3]):
            raise StripeQCError(
                f"Raw PA DWI volume {volume_index + 1} has unexpected shape "
                f"{volume.shape}; expected {tuple(shape[:3])}"
            )
        if not np.isfinite(volume).all():
            raise StripeQCError(
                f"Raw PA DWI volume {volume_index + 1} contains non-finite data"
            )

        sagittal_scores = np.empty(shape[0], dtype=np.float64)
        for sagittal_index in range(shape[0]):
            response = convolve2d(
                volume[sagittal_index, :, :], STRIPE_KERNEL, mode="full"
            )
            sagittal_scores[sagittal_index] = np.abs(response).sum(dtype=np.float64)
        a_si[volume_index] = sagittal_scores.sum(dtype=np.float64)
        peak_sagittal[volume_index] = int(np.argmax(sagittal_scores))

    shells = round_shells(values)
    c_si = np.empty_like(a_si)
    for shell in np.unique(shells):
        indices = np.flatnonzero(shells == shell)
        minimum = float(np.min(a_si[indices]))
        if not np.isfinite(minimum) or minimum <= 0:
            raise StripeQCError(
                f"Shell {int(shell)} has a non-finite or nonpositive aSI "
                f"minimum ({minimum}); verify that raw volumes contain usable signal"
            )
        c_si[indices] = a_si[indices] / minimum
    if not np.isfinite(c_si).all():
        raise StripeQCError("Within-shell cSI normalization produced non-finite values")
    return StripeMetrics(a_si, c_si, shells, peak_sagittal)


def classify_csi(value: float) -> str:
    """Classify one finite cSI as ``normal``, ``ambiguous``, or ``high``."""

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise StripeQCError("cSI classification requires a finite numeric value") from error
    if not np.isfinite(number):
        raise StripeQCError("cSI classification requires a finite numeric value")
    if number > _HIGH_THRESHOLD:
        return "high"
    if number >= _AMBIGUOUS_THRESHOLD:
        return "ambiguous"
    return "normal"


def decide_qc(metrics: StripeMetrics, ambiguous_reviewed: bool) -> QCDecision:
    """Apply the binding QC-gate precedence to stripe metrics."""

    if not isinstance(metrics, StripeMetrics):
        raise StripeQCError("metrics must be a StripeMetrics result")
    if not isinstance(ambiguous_reviewed, bool):
        raise StripeQCError("ambiguous_reviewed must be a boolean")

    high = tuple(int(index) for index in np.flatnonzero(metrics.c_si > _HIGH_THRESHOLD))
    ambiguous = tuple(
        int(index)
        for index in np.flatnonzero(
            (metrics.c_si >= _AMBIGUOUS_THRESHOLD)
            & (metrics.c_si <= _HIGH_THRESHOLD)
        )
    )
    if len(high) >= _EXCLUDE_HIGH_COUNT:
        return QCDecision("EXCLUDE", high, ambiguous, 20)
    if ambiguous and not ambiguous_reviewed:
        return QCDecision("HOLD_FOR_REVIEW", high, ambiguous, 21)
    if ambiguous:
        return QCDecision("INCLUDE_AFTER_REVIEW", high, ambiguous, 0)
    if high:
        return QCDecision("INCLUDE_WITH_FLAGS", high, ambiguous, 0)
    return QCDecision("INCLUDE", high, ambiguous, 0)


def run_stripe_qc(config: PipelineConfig, output_dir: Path) -> QCDecision:
    """Run raw PA stripe QC and write deterministic, subject-generic outputs."""

    if not isinstance(config, PipelineConfig):
        raise StripeQCError("config must be a PipelineConfig")
    destination = Path(output_dir)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise StripeQCError(
                f"QC output directory already exists unsafely: {destination}"
            )
        try:
            if any(destination.iterdir()):
                raise StripeQCError(
                    f"QC output directory already exists and is not empty: {destination}"
                )
        except OSError as error:
            raise StripeQCError(
                f"Cannot inspect QC output directory: {destination}"
            ) from error
    else:
        try:
            destination.mkdir(parents=True)
        except OSError as error:
            raise StripeQCError(
                f"Cannot create QC output directory: {destination}"
            ) from error

    image, bvals = _load_qc_inputs(config)
    metrics = compute_stripe_indices(image, bvals)
    if not np.any(metrics.shells == 0):
        raise StripeQCError(
            "Raw PA DWI must contain at least one b0 volume for the anatomy overview"
        )
    decision = decide_qc(metrics, config.analysis.ambiguous_qc_reviewed)

    _write_metrics(destination / "stripe_metrics.csv", bvals, metrics)
    _write_decision_json(
        destination / "stripe_decision.json", config, metrics, decision
    )
    _write_summary(
        destination / "automatic_summary.txt", config, metrics, decision
    )
    _plot_b0_overview(
        image,
        metrics.shells,
        destination / "00_raw_b0_anatomy_overview.png",
    )
    _plot_csi_by_volume(
        metrics,
        destination / "01_cSI_by_volume.png",
    )
    _plot_csi_by_shell(
        metrics,
        destination / "02_cSI_by_shell.png",
    )
    _plot_candidate_details(image, metrics, destination)
    _plot_all_volume_sheets(image, metrics, destination)
    return decision


def expected_stripe_detail_paths(
    config: PipelineConfig, output_dir: Path
) -> tuple[Path, ...]:
    """Return the exact detail-sheet paths that ``run_stripe_qc`` will write."""
    if not isinstance(config, PipelineConfig):
        raise StripeQCError("config must be a PipelineConfig")
    image, bvals = _load_qc_inputs(config)
    metrics = compute_stripe_indices(image, bvals)
    names = (
        *_candidate_detail_filenames(metrics),
        *_all_volume_filenames(int(image.shape[3])),
    )
    return tuple(Path(output_dir) / name for name in names)


def _load_qc_inputs(
    config: PipelineConfig,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    try:
        image = nib.load(config.dwi_pa, mmap=True)
    except (OSError, nib.filebasedimages.ImageFileError) as error:
        raise StripeQCError(f"Cannot read raw PA DWI: {config.dwi_pa}") from error
    try:
        bvals = np.loadtxt(config.bvals, dtype=float).reshape(-1)
    except (OSError, ValueError) as error:
        raise StripeQCError(f"Cannot read b-values: {config.bvals}") from error
    return image, bvals


def _candidate_indices(metrics: StripeMetrics) -> tuple[int, ...]:
    high = set(np.flatnonzero(metrics.c_si > _HIGH_THRESHOLD).tolist())
    ambiguous = set(
        np.flatnonzero(
            (metrics.c_si >= _AMBIGUOUS_THRESHOLD)
            & (metrics.c_si <= _HIGH_THRESHOLD)
        ).tolist()
    )
    top = set(np.argsort(metrics.c_si)[-min(12, metrics.c_si.size) :].tolist())
    return tuple(
        sorted(
            high | ambiguous | top,
            key=lambda index: (-float(metrics.c_si[index]), int(index)),
        )
    )


def _candidate_detail_filenames(metrics: StripeMetrics) -> tuple[str, ...]:
    candidate_count = len(_candidate_indices(metrics))
    return tuple(
        f"03_candidate_details_{first + 1:02d}_"
        f"{min(first + 6, candidate_count):02d}.png"
        for first in range(0, candidate_count, 6)
    )


def _all_volume_filenames(volume_count: int) -> tuple[str, ...]:
    return tuple(
        f"04_all_volumes_{first + 1:03d}_"
        f"{min(first + 36, volume_count):03d}.png"
        for first in range(0, volume_count, 36)
    )


def _readonly_float_array(value: object, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise StripeQCError(f"{name} must be a numeric one-dimensional array") from error
    if array.ndim != 1:
        raise StripeQCError(f"{name} must be a one-dimensional array")
    if not np.isfinite(array).all():
        raise StripeQCError(f"{name} must contain only finite values")
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


def _readonly_integer_array(value: object, name: str) -> np.ndarray:
    try:
        original = np.asarray(value)
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StripeQCError(f"{name} must be an integer one-dimensional array") from error
    if original.ndim != 1 or numeric.ndim != 1:
        raise StripeQCError(f"{name} must be a one-dimensional array")
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise StripeQCError(f"{name} must contain only finite integer values")
    array = np.array(numeric, dtype=np.int64, copy=True)
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


def _validated_indices(value: object, name: str) -> tuple[int, ...]:
    try:
        indices = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise StripeQCError(f"{name} must be an iterable of indices") from error
    if any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0
        for index in indices
    ):
        raise StripeQCError(f"{name} must contain nonnegative integer indices")
    normalized = tuple(int(index) for index in indices)
    if len(normalized) != len(set(normalized)) or normalized != tuple(sorted(normalized)):
        raise StripeQCError(f"{name} must contain unique indices in ascending order")
    return normalized


def _image_shape_and_data(image: object) -> tuple[tuple[int, ...], object]:
    if isinstance(image, np.ndarray):
        return tuple(int(length) for length in image.shape), image
    if isinstance(image, nib.spatialimages.SpatialImage):
        return tuple(int(length) for length in image.shape), image.dataobj
    raise StripeQCError("image must be a 4D NumPy array or NiBabel spatial image")


def _validated_bvals(bvals: object, n_volumes: int) -> np.ndarray:
    try:
        values = np.asarray(bvals, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise StripeQCError("b-values must be a finite numeric sequence") from error
    if values.size != n_volumes:
        raise StripeQCError(
            f"b-value count {values.size} does not match PA volume count {n_volumes}"
        )
    if not np.isfinite(values).all():
        raise StripeQCError("b-values must contain only finite values")
    if np.any(values < 0):
        raise StripeQCError("b-values must be nonnegative")
    return values


def _write_metrics(path: Path, bvals: np.ndarray, metrics: StripeMetrics) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "volume_index_zero_based",
                "volume_number_one_based",
                "b_value",
                "nominal_shell",
                "a_si",
                "c_si",
                "classification",
                "peak_sagittal_index_zero_based",
                "peak_sagittal_number_one_based",
            ]
        )
        for index in range(metrics.c_si.size):
            writer.writerow(
                [
                    index,
                    index + 1,
                    f"{float(bvals[index]):.8g}",
                    int(metrics.shells[index]),
                    f"{float(metrics.a_si[index]):.12g}",
                    f"{float(metrics.c_si[index]):.17g}",
                    classify_csi(float(metrics.c_si[index])),
                    int(metrics.peak_sagittal[index]),
                    int(metrics.peak_sagittal[index]) + 1,
                ]
            )


def _write_decision_json(
    path: Path,
    config: PipelineConfig,
    metrics: StripeMetrics,
    decision: QCDecision,
) -> None:
    maximum_index = int(np.argmax(metrics.c_si))
    unique_shells, counts = np.unique(metrics.shells, return_counts=True)
    normal_count = int(
        np.count_nonzero(metrics.c_si < _AMBIGUOUS_THRESHOLD)
    )
    payload = {
        "subject_id": config.subject_id,
        "method": _METHOD_NAME,
        "thresholds": {
            "ambiguous_min_inclusive": _AMBIGUOUS_THRESHOLD,
            "high_min_exclusive": _HIGH_THRESHOLD,
            "exclude_high_volume_count": _EXCLUDE_HIGH_COUNT,
        },
        "decision": decision.status,
        "exit_code": decision.exit_code,
        "ambiguous_reviewed": config.analysis.ambiguous_qc_reviewed,
        "volume_counts": {
            "total": int(metrics.c_si.size),
            "normal": normal_count,
            "ambiguous": len(decision.ambiguous_indices),
            "high": len(decision.high_indices),
        },
        "flagged_indices_zero_based": {
            "high": list(decision.high_indices),
            "ambiguous": list(decision.ambiguous_indices),
        },
        "flagged_volume_numbers_one_based": {
            "high": [index + 1 for index in decision.high_indices],
            "ambiguous": [index + 1 for index in decision.ambiguous_indices],
        },
        "maximum_csi": {
            "value": float(metrics.c_si[maximum_index]),
            "volume_index_zero_based": maximum_index,
            "volume_number_one_based": maximum_index + 1,
        },
        "shell_counts": {
            str(int(shell)): int(count)
            for shell, count in zip(unique_shells, counts, strict=True)
        },
        "cohort_fsi": (
            "Cohort fSI was not computed because this is a single-subject "
            "technical QC screen."
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_summary(
    path: Path,
    config: PipelineConfig,
    metrics: StripeMetrics,
    decision: QCDecision,
) -> None:
    maximum_index = int(np.argmax(metrics.c_si))
    unique_shells, counts = np.unique(metrics.shells, return_counts=True)
    shell_counts = {
        int(shell): int(count)
        for shell, count in zip(unique_shells, counts, strict=True)
    }
    lines = [
        f"Subject: {config.subject_id}",
        "Single-subject pre-denoise motion/stripe QC",
        f"Method: {_METHOD_NAME}",
        f"Volumes: {metrics.c_si.size}",
        (
            "High cSI > 1.25: "
            f"{len(decision.high_indices)}; zero-based {list(decision.high_indices)}; "
            f"one-based {[index + 1 for index in decision.high_indices]}"
        ),
        (
            "Ambiguous 1.15 <= cSI <= 1.25: "
            f"{len(decision.ambiguous_indices)}; "
            f"zero-based {list(decision.ambiguous_indices)}; "
            f"one-based {[index + 1 for index in decision.ambiguous_indices]}"
        ),
        (
            f"Maximum cSI: {float(metrics.c_si[maximum_index]):.6f} at "
            f"zero-based index {maximum_index} (one-based volume {maximum_index + 1})"
        ),
        f"Automatic status: {decision.status}",
        f"Exit code: {decision.exit_code}",
        f"Shell counts: {shell_counts}",
        "Cohort fSI was not computed for this single-subject technical QC screen.",
        "This technical QC result is not a clinical image interpretation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _robust_limits(array: np.ndarray) -> tuple[float, float]:
    values = np.asarray(array)[np.isfinite(array)]
    if values.size == 0:
        return 0.0, 1.0
    lower = float(np.percentile(values, 1.0))
    upper = float(np.percentile(values, 99.5))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _volume(image: nib.spatialimages.SpatialImage, index: int) -> np.ndarray:
    return np.asarray(image.dataobj[..., index], dtype=np.float32)


def _plot_b0_overview(
    image: nib.spatialimages.SpatialImage,
    shells: np.ndarray,
    path: Path,
) -> None:
    b0_indices = np.flatnonzero(shells == 0)
    mean_b0 = np.zeros(tuple(image.shape[:3]), dtype=np.float64)
    for index in b0_indices:
        mean_b0 += _volume(image, int(index))
    mean_b0 /= b0_indices.size
    centers = tuple(length // 2 for length in mean_b0.shape)
    views = (
        np.rot90(mean_b0[centers[0], :, :]),
        np.rot90(mean_b0[:, centers[1], :]),
        np.rot90(mean_b0[:, :, centers[2]]),
    )
    vmin, vmax = _robust_limits(mean_b0)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, view, title in zip(
        axes, views, ("Sagittal", "Coronal", "Axial"), strict=True
    ):
        axis.imshow(view, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Raw pre-denoise mean b0 ({b0_indices.size} volumes)")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_csi_by_volume(metrics: StripeMetrics, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for color_index, shell in enumerate(np.unique(metrics.shells)):
        indices = np.flatnonzero(metrics.shells == shell)
        label = "b=0" if shell == 0 else f"b={int(shell)}"
        axis.scatter(
            indices + 1,
            metrics.c_si[indices],
            s=24,
            alpha=0.8,
            color=colors(color_index % 10),
            label=label,
        )
    high = np.flatnonzero(metrics.c_si > _HIGH_THRESHOLD)
    ambiguous = np.flatnonzero(
        (metrics.c_si >= _AMBIGUOUS_THRESHOLD)
        & (metrics.c_si <= _HIGH_THRESHOLD)
    )
    axis.scatter(
        high + 1,
        metrics.c_si[high],
        s=90,
        facecolors="none",
        edgecolors="red",
        linewidths=1.8,
    )
    axis.scatter(
        ambiguous + 1,
        metrics.c_si[ambiguous],
        s=80,
        facecolors="none",
        edgecolors="darkorange",
        linewidths=1.8,
    )
    for index in np.concatenate((high, ambiguous)):
        axis.annotate(
            str(int(index) + 1),
            (int(index) + 1, float(metrics.c_si[index])),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhspan(
        _AMBIGUOUS_THRESHOLD, _HIGH_THRESHOLD, color="orange", alpha=0.09
    )
    axis.axhline(
        _HIGH_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="cSI > 1.25: high",
    )
    axis.axhline(
        _AMBIGUOUS_THRESHOLD,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
        label="cSI 1.15–1.25: ambiguous",
    )
    axis.set_xlabel("DWI volume number (one-based)")
    axis.set_ylabel("Corrected stripe index (cSI)")
    axis.set_title("Pre-denoise motion/stripe screening")
    axis.grid(axis="y", color="0.85", linewidth=0.6)
    axis.legend(loc="upper right", fontsize=8, ncol=3)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_csi_by_shell(metrics: StripeMetrics, path: Path) -> None:
    unique_shells = np.unique(metrics.shells)
    values = [metrics.c_si[metrics.shells == shell] for shell in unique_shells]
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.boxplot(
        values,
        tick_labels=[
            "b0" if shell == 0 else str(int(shell)) for shell in unique_shells
        ],
        showfliers=False,
    )
    generator = np.random.default_rng(0)
    for position, shell_values in enumerate(values, 1):
        jitter = generator.uniform(-0.13, 0.13, shell_values.size)
        axis.scatter(position + jitter, shell_values, s=20, alpha=0.65)
    axis.axhspan(
        _AMBIGUOUS_THRESHOLD, _HIGH_THRESHOLD, color="orange", alpha=0.09
    )
    axis.axhline(_HIGH_THRESHOLD, color="red", linestyle="--", linewidth=1.2)
    axis.axhline(
        _AMBIGUOUS_THRESHOLD,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
    )
    axis.set_xlabel("Nominal b-value shell (s/mm²)")
    axis.set_ylabel("Corrected stripe index (cSI)")
    axis.set_title("cSI distribution within each diffusion shell")
    axis.grid(axis="y", color="0.85", linewidth=0.6)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_candidate_details(
    image: nib.spatialimages.SpatialImage,
    metrics: StripeMetrics,
    output_dir: Path,
) -> list[Path]:
    candidates = _candidate_indices(metrics)
    filenames = _candidate_detail_filenames(metrics)
    paths: list[Path] = []
    per_sheet = 6
    for sheet_index, first in enumerate(range(0, len(candidates), per_sheet)):
        selection = candidates[first : first + per_sheet]
        figure, axes = plt.subplots(
            len(selection),
            2,
            figsize=(12, 3.1 * len(selection)),
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        for row, volume_index in enumerate(selection):
            sagittal_index = int(metrics.peak_sagittal[volume_index])
            raw = np.asarray(
                image.dataobj[sagittal_index, :, :, volume_index],
                dtype=np.float64,
            )
            response = np.abs(convolve2d(raw, STRIPE_KERNEL, mode="full"))
            vmin, vmax = _robust_limits(raw)
            response_max = float(np.percentile(response, 99.5))
            if response_max <= 0:
                response_max = 1.0
            axes[row, 0].imshow(np.rot90(raw), cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, 1].imshow(
                np.rot90(response), cmap="magma", vmin=0, vmax=response_max
            )
            axes[row, 0].set_title(
                f"V{volume_index + 1}, b={int(metrics.shells[volume_index])}, "
                f"x={sagittal_index + 1}, cSI={metrics.c_si[volume_index]:.4f} "
                f"({classify_csi(float(metrics.c_si[volume_index]))})"
            )
            axes[row, 1].set_title("Absolute stripe-filter response")
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
        path = output_dir / filenames[sheet_index]
        figure.suptitle("Highest-cSI raw volumes (one-based labels)", fontsize=15)
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_all_volume_sheets(
    image: nib.spatialimages.SpatialImage,
    metrics: StripeMetrics,
    output_dir: Path,
) -> list[Path]:
    central_sagittal = image.shape[0] // 2
    paths: list[Path] = []
    per_sheet = 36
    filenames = _all_volume_filenames(int(image.shape[3]))
    for sheet_index, first in enumerate(range(0, image.shape[3], per_sheet)):
        last = min(first + per_sheet, image.shape[3])
        figure, axes = plt.subplots(
            6, 6, figsize=(18, 18), constrained_layout=True
        )
        flat_axes = axes.ravel()
        for axis, volume_index in zip(
            flat_axes, range(first, last), strict=False
        ):
            view = np.rot90(
                np.asarray(
                    image.dataobj[central_sagittal, :, :, volume_index],
                    dtype=np.float32,
                )
            )
            vmin, vmax = _robust_limits(view)
            axis.imshow(view, cmap="gray", vmin=vmin, vmax=vmax)
            classification = classify_csi(float(metrics.c_si[volume_index]))
            color = {
                "normal": "black",
                "ambiguous": "darkorange",
                "high": "red",
            }[classification]
            axis.set_title(
                f"V{volume_index + 1}  b{int(metrics.shells[volume_index])}\n"
                f"cSI {metrics.c_si[volume_index]:.3f}",
                fontsize=8,
                color=color,
            )
            axis.axis("off")
        for axis in flat_axes[last - first :]:
            axis.axis("off")
        figure.suptitle(
            f"Raw central sagittal quicklook — volumes {first + 1}–{last} "
            "(one-based)",
            fontsize=16,
        )
        path = output_dir / filenames[sheet_index]
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
