"""Deterministic 48-ROI and global quantitative summaries."""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import os
import stat
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
from .resources import (
    JHU_IMAGE_SHA256,
    JHU_XML_SHA256,
    ResourceValidationError,
    _parse_label_names,
)


CANONICAL_METRICS = (
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
COUNT_FIELDS = (
    "atlas_labeled_voxel_count",
    "after_brain_mask_voxel_count",
    "after_noddi_success_voxel_count",
    "after_fiso_finite_voxel_count",
    "after_fiso_threshold_voxel_count",
    "common_mask_voxel_count",
)
_EXPECTED_LABELS = tuple(range(1, 49))
_MASK_DEFINITION = (
    "warped_jhu_label_in_1_to_48",
    "cleaned_bet_brain_mask",
    "noddi_error_code_equals_0",
    "finite_noddi_fiso",
    "noddi_fiso_less_than_or_equal_to_0.9",
    "finite_all_canonical_metrics",
)


class SummaryError(ValueError):
    """Raised when summary inputs or destinations violate the scientific contract."""


@dataclass(frozen=True)
class SummaryContext:
    """Validated, immutable assignment of all inputs for one subject summary."""

    config: PipelineConfig
    warped_atlas: Path
    brain_mask: Path
    metric_maps: Mapping[str, Path]
    noddi_error_codes: Path
    atlas_xml: Path
    atlas_provenance: Path
    output_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.config, PipelineConfig):
            raise SummaryError("summary config must be a PipelineConfig")
        try:
            supplied_metrics = dict(self.metric_maps)
        except (TypeError, ValueError) as error:
            raise SummaryError("metric_maps must be a canonical path mapping") from error
        supplied_keys = set(supplied_metrics)
        expected_keys = set(CANONICAL_METRICS)
        if supplied_keys != expected_keys:
            missing = sorted(expected_keys - supplied_keys)
            extra = sorted(supplied_keys - expected_keys)
            raise SummaryError(
                f"metric_maps must contain exact canonical metric keys; "
                f"missing={missing}, extra={extra}"
            )
        normalized = {
            metric: Path(supplied_metrics[metric]) for metric in CANONICAL_METRICS
        }
        object.__setattr__(self, "warped_atlas", Path(self.warped_atlas))
        object.__setattr__(self, "brain_mask", Path(self.brain_mask))
        object.__setattr__(self, "metric_maps", MappingProxyType(normalized))
        object.__setattr__(self, "noddi_error_codes", Path(self.noddi_error_codes))
        object.__setattr__(self, "atlas_xml", Path(self.atlas_xml))
        object.__setattr__(self, "atlas_provenance", Path(self.atlas_provenance))
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        _validate_path_assignment(self)
        with _InputSnapshots(self) as snapshots:
            _load_summary_data(self, snapshots)


@dataclass(frozen=True)
class SummaryOutputs:
    """Paths to the three newly created deterministic summary products."""

    roi_csv: Path
    global_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class _SummaryData:
    atlas: np.ndarray
    brain: np.ndarray
    errors: np.ndarray
    metrics: Mapping[str, np.ndarray]
    affine: np.ndarray
    label_names: Mapping[int, str]
    provenance: Mapping[str, object]
    warped_atlas_sha256: str


@dataclass(frozen=True)
class _HeldInput:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    digest: str
    snapshot_path: Path


@dataclass(frozen=True)
class _CreatedOutput:
    name: str
    identity: tuple[int, int]
    size: int
    digest: str


class _InputSnapshots:
    """Hold original inputs open and expose immutable byte-identical snapshots."""

    def __init__(self, context: SummaryContext) -> None:
        self.context = context
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._held: list[_HeldInput] = []
        self.paths: Mapping[Path, Path] = MappingProxyType({})
        self.hashes: Mapping[Path, str] = MappingProxyType({})

    def __enter__(self) -> "_InputSnapshots":
        self._temporary = tempfile.TemporaryDirectory(prefix="dmri-summary-inputs-")
        snapshot_root = Path(self._temporary.name)
        try:
            snapshots: dict[Path, Path] = {}
            hashes: dict[Path, str] = {}
            for index, path in enumerate(_summary_inputs(self.context)):
                descriptor = _open_read_no_symlink(path)
                try:
                    metadata = os.fstat(descriptor)
                    identity = (metadata.st_dev, metadata.st_ino)
                    suffix = "".join(path.suffixes) or ".bin"
                    snapshot_path = snapshot_root / f"{index:02d}{suffix}"
                    digest = _copy_descriptor(descriptor, snapshot_path)
                    if _hash_descriptor(descriptor) != digest:
                        raise SummaryError(
                            f"summary input changed while snapshotting: {path.name}"
                        )
                    final_metadata = os.fstat(descriptor)
                    if (final_metadata.st_dev, final_metadata.st_ino) != identity:
                        raise SummaryError(
                            f"summary input identity changed while snapshotting: {path.name}"
                        )
                except Exception:
                    os.close(descriptor)
                    raise
                held = _HeldInput(
                    path=path,
                    descriptor=descriptor,
                    identity=identity,
                    digest=digest,
                    snapshot_path=snapshot_path,
                )
                self._held.append(held)
                snapshots[path] = snapshot_path
                hashes[path] = digest
            self.paths = MappingProxyType(snapshots)
            self.hashes = MappingProxyType(hashes)
            self.verify()
            return self
        except Exception:
            self.close()
            raise

    def verify(self) -> None:
        for held in self._held:
            if _sha256(held.snapshot_path) != held.digest:
                raise SummaryError("immutable summary input snapshot changed")
            metadata = os.fstat(held.descriptor)
            if (metadata.st_dev, metadata.st_ino) != held.identity:
                raise SummaryError("summary input changed during summary computation")
            if _hash_descriptor(held.descriptor) != held.digest:
                raise SummaryError("summary input changed during summary computation")
            current_descriptor = _open_read_no_symlink(held.path)
            try:
                current_metadata = os.fstat(current_descriptor)
                current_identity = (current_metadata.st_dev, current_metadata.st_ino)
                if current_identity != held.identity:
                    raise SummaryError(
                        "summary input changed during summary computation"
                    )
                if _hash_descriptor(current_descriptor) != held.digest:
                    raise SummaryError(
                        "summary input changed during summary computation"
                    )
            finally:
                os.close(current_descriptor)

    def close(self) -> None:
        for held in self._held:
            try:
                os.close(held.descriptor)
            except OSError:
                pass
        self._held.clear()
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except OSError:
                pass
            self._temporary = None

    def __exit__(self, *_args: object) -> None:
        self.close()


def build_summary_mask(context: SummaryContext) -> np.ndarray:
    """Return the defensive, read-only common finite-voxel summary mask."""
    _require_context(context)
    with _InputSnapshots(context) as snapshots:
        data = _load_summary_data(context, snapshots)
        snapshots.verify()
    mask, _ = _mask_and_counts(data)
    defensive = np.array(mask, dtype=bool, copy=True)
    defensive.setflags(write=False)
    return defensive


def summarize_subject(context: SummaryContext) -> SummaryOutputs:
    """Create one 48-row ROI table, one global row, and deterministic JSON."""
    _require_context(context)
    _validate_path_assignment(context)
    _preflight_outputs(context)
    with _InputSnapshots(context) as snapshots:
        data = _load_summary_data(context, snapshots)
        mask, counts = _mask_and_counts(data)

        roi_rows: list[list[object]] = []
        voxel_total = 0
        for label_id in _EXPECTED_LABELS:
            roi_mask = mask & (data.atlas == label_id)
            voxel_count = int(np.count_nonzero(roi_mask))
            if voxel_count == 0:
                raise SummaryError(
                    f"JHU label {label_id} has zero valid voxels in the common summary mask"
                )
            voxel_total += voxel_count
            row: list[object] = [
                context.config.subject_id,
                label_id,
                data.label_names[label_id],
                voxel_count,
            ]
            row.extend(
                _aggregate(data.metrics[metric][roi_mask], metric)
                for metric in CANONICAL_METRICS
            )
            roi_rows.append(row)
        if voxel_total != counts["common_mask_voxel_count"]:
            raise SummaryError("common-mask count does not equal the 48 ROI voxel counts")

        global_values = [
            _aggregate(data.metrics[metric][mask], metric)
            for metric in CANONICAL_METRICS
        ]
        _require_finite_outputs(roi_rows, global_values)

        roi_header = (
            "subject_id",
            "label_id",
            "label_name",
            "voxel_count",
            *CANONICAL_METRICS,
        )
        global_header = ("subject_id", *COUNT_FIELDS, *CANONICAL_METRICS)
        global_row: list[object] = [
            context.config.subject_id,
            *(counts[field] for field in COUNT_FIELDS),
            *global_values,
        ]
        roi_text = _csv_text(roi_header, roi_rows)
        global_text = _csv_text(global_header, [global_row])
        summary_payload = _summary_payload(context, data, counts, roi_rows)
        try:
            json_text = (
                json.dumps(
                    summary_payload, allow_nan=False, indent=2, ensure_ascii=False
                )
                + "\n"
            )
        except (TypeError, ValueError) as error:
            raise SummaryError(
                "summary JSON contains a non-finite or unsupported value"
            ) from error

        outputs = _output_paths(context)
        _write_outputs_no_clobber(
            context,
            {
                outputs.roi_csv: roi_text,
                outputs.global_csv: global_text,
                outputs.summary_json: json_text,
            },
            snapshots,
        )
        return outputs


def _require_context(context: SummaryContext) -> None:
    if not isinstance(context, SummaryContext):
        raise SummaryError("context must be a SummaryContext")


def _validate_path_assignment(context: SummaryContext) -> None:
    subject_root = context.config.subject_output
    subject_paths = [
        context.warped_atlas,
        context.brain_mask,
        context.noddi_error_codes,
        *context.metric_maps.values(),
    ]
    all_inputs = [*subject_paths, context.atlas_xml, context.atlas_provenance]
    identities: dict[tuple[int, int], Path] = {}
    for path in all_inputs:
        _require_regular_input(path)
        identity = _file_identity(path)
        if identity in identities:
            raise SummaryError(
                f"summary input alias is not allowed: {path.name} aliases "
                f"{identities[identity].name}"
            )
        identities[identity] = path
    for path in subject_paths:
        _relative_to_subject(path, subject_root)

    output = context.output_directory
    _reject_parent_traversal(output, "summary output directory")
    _reject_symlink_components(output, "summary output directory")
    _relative_to_subject(output, subject_root, require_exists=False)
    if os.path.lexists(output):
        try:
            mode = output.lstat().st_mode
        except OSError as error:
            raise SummaryError("cannot inspect summary output directory") from error
        if stat.S_ISLNK(mode):
            raise SummaryError("summary output directory must not be a symlink")
        if not stat.S_ISDIR(mode):
            raise SummaryError("summary output directory must be a directory")


def _require_regular_input(path: Path) -> None:
    _reject_parent_traversal(path, "summary input")
    _reject_symlink_components(path, f"summary input {path.name}")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise SummaryError(f"summary input is not readable: {path.name}") from error
    if not stat.S_ISREG(mode):
        raise SummaryError(f"summary input must be a regular file: {path.name}")


def _reject_parent_traversal(path: Path, label: str) -> None:
    if ".." in path.parts:
        raise SummaryError(f"{label} must not contain parent traversal")


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise SummaryError(f"{label} must not contain symlink components")


def _relative_to_subject(
    path: Path, subject_root: Path, *, require_exists: bool = True
) -> Path:
    try:
        resolved_path = path.resolve(strict=require_exists)
        resolved_root = subject_root.resolve(strict=False)
        relative = resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise SummaryError(
            f"summary subject path must stay within subject output: {path.name}"
        ) from error
    if not relative.parts:
        raise SummaryError("summary path must name an item within subject output")
    return relative


def _file_identity(path: Path) -> tuple[int, int]:
    try:
        details = path.stat(follow_symlinks=False)
    except OSError as error:
        raise SummaryError(f"cannot inspect summary input: {path.name}") from error
    return details.st_dev, details.st_ino


def _load_summary_data(
    context: SummaryContext, snapshots: _InputSnapshots
) -> _SummaryData:
    atlas, affine = _load_nifti(
        snapshots.paths[context.warped_atlas], "warped atlas"
    )
    _require_finite(atlas, "warped atlas")
    _require_integral(atlas, "warped atlas")
    labels = tuple(int(value) for value in np.unique(atlas) if value != 0)
    if labels != _EXPECTED_LABELS or np.any(atlas < 0) or np.any(atlas > 48):
        raise SummaryError(
            "warped atlas must have nonzero labels exactly 1 through 48"
        )

    brain, brain_affine = _load_nifti(
        snapshots.paths[context.brain_mask], "cleaned brain mask"
    )
    _require_grid(brain, brain_affine, atlas.shape, affine, "cleaned brain mask")
    _require_finite(brain, "cleaned brain mask")
    _require_integral(brain, "cleaned brain mask")
    if not np.isin(brain, (0.0, 1.0)).all():
        raise SummaryError("cleaned brain mask must contain only 0 and 1")

    errors, errors_affine = _load_nifti(
        snapshots.paths[context.noddi_error_codes], "NODDI error-code map"
    )
    _require_grid(
        errors, errors_affine, atlas.shape, affine, "NODDI error-code map"
    )
    _require_finite(errors, "NODDI error-code map")
    _require_integral(errors, "NODDI error-code map")

    metrics: dict[str, np.ndarray] = {}
    for metric in CANONICAL_METRICS:
        values, metric_affine = _load_nifti(
            snapshots.paths[context.metric_maps[metric]], metric
        )
        _require_grid(values, metric_affine, atlas.shape, affine, metric)
        metrics[metric] = values

    try:
        names = _parse_label_names(snapshots.paths[context.atlas_xml])
    except ResourceValidationError as error:
        raise SummaryError(f"invalid atlas XML: {error}") from error
    if snapshots.hashes[context.atlas_xml] != JHU_XML_SHA256:
        raise SummaryError("atlas XML SHA-256 does not match the accepted resource")
    provenance = _load_provenance(
        snapshots.paths[context.atlas_provenance], names
    )
    return _SummaryData(
        atlas=atlas,
        brain=brain,
        errors=errors,
        metrics=MappingProxyType(metrics),
        affine=affine,
        label_names=MappingProxyType(names),
        provenance=MappingProxyType(provenance),
        warped_atlas_sha256=snapshots.hashes[context.warped_atlas],
    )


def _load_nifti(path: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        image = nib.load(path)
        if len(image.shape) != 3:
            raise SummaryError(f"{label} must be a numeric 3D NIfTI")
        dtype = image.get_data_dtype()
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise SummaryError(f"{label} must be a numeric 3D NIfTI")
        affine = np.asarray(image.affine, dtype=np.float64)
        if not np.isfinite(affine).all():
            raise SummaryError(f"{label} affine must be finite")
        values = np.asarray(image.dataobj, dtype=np.float64)
    except SummaryError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        ImageFileError,
        HeaderDataError,
        ImageDataError,
    ) as error:
        raise SummaryError(f"cannot read {label} NIfTI") from error
    return values, affine


def _require_grid(
    values: np.ndarray,
    affine: np.ndarray,
    expected_shape: tuple[int, ...],
    expected_affine: np.ndarray,
    label: str,
) -> None:
    if values.shape != expected_shape or not np.allclose(
        affine, expected_affine, atol=1e-5, rtol=0.0
    ):
        raise SummaryError(f"{label} must use the warped atlas grid and affine")


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise SummaryError(f"{label} must be globally finite")


def _require_integral(values: np.ndarray, label: str) -> None:
    if not np.equal(values, np.rint(values)).all():
        raise SummaryError(f"{label} must be exactly integer-valued")


def _load_provenance(path: Path, names: Mapping[int, str]) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError("cannot read atlas provenance JSON") from error
    if not isinstance(payload, dict):
        raise SummaryError("atlas provenance must be a JSON object")
    expected_source = {"component": "data_atlases", "tag": "fsl-5_0_4"}
    expected_files = {
        "JHU-ICBM-labels-2mm.nii.gz": {"sha256": JHU_IMAGE_SHA256},
        "JHU-labels.xml": {"sha256": JHU_XML_SHA256},
    }
    expected_mapping = {str(index): names[index] for index in range(49)}
    if payload.get("source") != expected_source:
        raise SummaryError("atlas provenance has the wrong FSL source tag")
    if payload.get("files") != expected_files:
        raise SummaryError("atlas provenance resource hashes do not match")
    if payload.get("nonzero_labels") != list(_EXPECTED_LABELS):
        raise SummaryError("atlas provenance must record labels 1 through 48")
    if payload.get("nearest_neighbour_only") is not True:
        raise SummaryError("atlas provenance must require nearest-neighbour warping")
    if payload.get("label_mapping") != expected_mapping:
        raise SummaryError("atlas provenance true label mapping does not match XML")
    return payload


def _mask_and_counts(
    data: _SummaryData,
) -> tuple[np.ndarray, dict[str, int]]:
    mask = (data.atlas >= 1) & (data.atlas <= 48)
    counts = {COUNT_FIELDS[0]: int(np.count_nonzero(mask))}
    mask &= data.brain == 1
    counts[COUNT_FIELDS[1]] = int(np.count_nonzero(mask))
    mask &= data.errors == 0
    counts[COUNT_FIELDS[2]] = int(np.count_nonzero(mask))
    fiso = data.metrics["NODDI_FISO"]
    mask &= np.isfinite(fiso)
    counts[COUNT_FIELDS[3]] = int(np.count_nonzero(mask))
    mask &= fiso <= 0.9
    counts[COUNT_FIELDS[4]] = int(np.count_nonzero(mask))
    for metric in CANONICAL_METRICS:
        mask &= np.isfinite(data.metrics[metric])
    counts[COUNT_FIELDS[5]] = int(np.count_nonzero(mask))
    ordered_counts = [counts[field] for field in COUNT_FIELDS]
    if any(
        later > earlier for earlier, later in zip(ordered_counts, ordered_counts[1:])
    ):
        raise SummaryError("common-mask cumulative counts must be monotonic")
    return mask, counts


def _aggregate(values: np.ndarray, metric: str) -> float:
    if values.size == 0:
        raise SummaryError(f"{metric} has zero valid voxels")
    if metric.endswith(("_MK", "_AK", "_RK")):
        result = float(np.median(values))
    else:
        result = float(np.mean(values))
    if not np.isfinite(result):
        raise SummaryError(f"{metric} aggregation is non-finite")
    return result


def _require_finite_outputs(
    roi_rows: list[list[object]], global_values: list[float]
) -> None:
    numeric = [
        float(value)
        for row in roi_rows
        for value in row[3:]
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    numeric.extend(global_values)
    if not np.isfinite(numeric).all():
        raise SummaryError("summary tables must not contain NaN or infinity")


def _csv_text(header: tuple[str, ...], rows: list[list[object]]) -> str:
    destination = io.StringIO(newline="")
    writer = csv.writer(destination, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return destination.getvalue()


def _summary_payload(
    context: SummaryContext,
    data: _SummaryData,
    counts: Mapping[str, int],
    roi_rows: list[list[object]],
) -> dict[str, object]:
    subject_root = context.config.subject_output
    brain_mask = data.brain == 1
    histogram_values, histogram_counts = np.unique(
        data.errors[brain_mask].astype(np.int64), return_counts=True
    )
    histogram = [
        {"error_code": int(code), "voxel_count": int(count)}
        for code, count in zip(histogram_values, histogram_counts)
    ]
    aggregation_rules = {
        metric: (
            "median" if metric.endswith(("_MK", "_AK", "_RK")) else "mean"
        )
        for metric in CANONICAL_METRICS
    }
    output_paths = _output_paths(context)
    atlas_mapping = [
        {"index": index, "name": data.label_names[index]} for index in range(49)
    ]
    return {
        "subject_id": context.config.subject_id,
        "atlas": {
            "source": data.provenance["source"],
            "resource_files": data.provenance["files"],
            "warped_atlas_path": str(
                _relative_to_subject(context.warped_atlas, subject_root)
            ),
            "warped_atlas_sha256": data.warped_atlas_sha256,
            "label_mapping": atlas_mapping,
            "nonzero_labels": list(_EXPECTED_LABELS),
            "warping_interpolation": "nearest-neighbour only",
        },
        "input_paths": {
            "brain_mask": str(_relative_to_subject(context.brain_mask, subject_root)),
            "noddi_error_codes": str(
                _relative_to_subject(context.noddi_error_codes, subject_root)
            ),
        },
        "metric_maps": {
            metric: str(
                _relative_to_subject(context.metric_maps[metric], subject_root)
            )
            for metric in CANONICAL_METRICS
        },
        "common_mask": {
            "definition": list(_MASK_DEFINITION),
            "counts": {field: counts[field] for field in COUNT_FIELDS},
            "technical_filter_only": True,
        },
        "aggregation_rules": aggregation_rules,
        "noddi_error_code_histogram": {
            "scope": "cleaned_brain_mask",
            "bins": histogram,
        },
        "roi_voxel_counts": {
            str(int(row[1])): int(row[3]) for row in roi_rows
        },
        "outputs": {
            "roi_csv": str(
                _relative_to_subject(
                    output_paths.roi_csv, subject_root, require_exists=False
                )
            ),
            "global_csv": str(
                _relative_to_subject(
                    output_paths.global_csv, subject_root, require_exists=False
                )
            ),
            "summary_json": str(
                _relative_to_subject(
                    output_paths.summary_json, subject_root, require_exists=False
                )
            ),
        },
    }


def _output_paths(context: SummaryContext) -> SummaryOutputs:
    subject = context.config.subject_id
    return SummaryOutputs(
        roi_csv=context.output_directory / f"{subject}_JHU_48ROI_metrics.csv",
        global_csv=context.output_directory / f"{subject}_global_metrics.csv",
        summary_json=context.output_directory / f"{subject}_summary.json",
    )


def _write_outputs_no_clobber(
    context: SummaryContext,
    documents: Mapping[Path, str],
    snapshots: _InputSnapshots,
) -> None:
    _verify_input_hashes(context, snapshots.hashes)
    snapshots.verify()
    _reject_symlink_components(context.output_directory, "summary output directory")
    directory_fd = _open_directory_no_symlink(
        context.output_directory, create=True
    )
    directory_identity = _descriptor_identity(directory_fd)
    created_files: list[_CreatedOutput] = []
    try:
        for destination, content in documents.items():
            if destination.parent != context.output_directory:
                raise SummaryError("summary outputs must share the assigned directory")
            _require_missing_at(directory_fd, destination.name)
            descriptor: int | None = None
            try:
                descriptor = _open_exclusive_at(directory_fd, destination.name)
                intended_bytes = content.encode("utf-8")
                created_files.append(
                    _CreatedOutput(
                        name=destination.name,
                        identity=_descriptor_identity(descriptor),
                        size=len(intended_bytes),
                        digest=hashlib.sha256(intended_bytes).hexdigest(),
                    )
                )
                with os.fdopen(
                    descriptor, "w", encoding="utf-8", newline="", closefd=True
                ) as handle:
                    descriptor = None
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as error:
                raise SummaryError(
                    f"summary output already exists: {destination.name}"
                ) from error
            except OSError as error:
                raise SummaryError(
                    f"cannot write summary output: {destination.name}"
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        snapshots.verify()
        _require_directory_binding(context.output_directory, directory_identity)
        os.fsync(directory_fd)
        snapshots.verify()
        _require_directory_binding(context.output_directory, directory_identity)
        snapshots.verify()
        _require_commit_state(
            directory_fd,
            created_files,
            context.output_directory,
            directory_identity,
        )
    except (OSError, SummaryError) as error:
        for created in reversed(created_files):
            _rollback_created_at(
                directory_fd, created.name, created.identity
            )
        if isinstance(error, SummaryError):
            raise
        raise SummaryError("cannot commit summary outputs") from error
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _preflight_outputs(context: SummaryContext) -> None:
    for destination in (
        _output_paths(context).roi_csv,
        _output_paths(context).global_csv,
        _output_paths(context).summary_json,
    ):
        if os.path.lexists(destination):
            raise SummaryError(f"summary output already exists: {destination.name}")


def _summary_inputs(context: SummaryContext) -> tuple[Path, ...]:
    return (
        context.warped_atlas,
        context.brain_mask,
        context.noddi_error_codes,
        context.atlas_xml,
        context.atlas_provenance,
        *(context.metric_maps[metric] for metric in CANONICAL_METRICS),
    )


def _verify_input_hashes(
    context: SummaryContext, expected: Mapping[Path, str]
) -> None:
    _validate_path_assignment(context)
    current = {path: _sha256(path) for path in _summary_inputs(context)}
    if current != dict(expected):
        raise SummaryError("summary input changed during summary computation")


def _open_directory_no_symlink(path: Path, *, create: bool) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError(
            errno.ENOTSUP,
            "platform lacks safe no-follow output traversal support",
        )
    absolute = Path(os.path.abspath(path))
    if len(absolute.parts) < 2:
        raise OSError(errno.EINVAL, "summary output directory is invalid")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    directory_fd = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:]:
            try:
                metadata = os.stat(
                    component, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                metadata = os.stat(
                    component, dir_fd=directory_fd, follow_symlinks=False
                )
            if stat.S_ISLNK(metadata.st_mode):
                raise OSError(
                    errno.ELOOP,
                    f"symlink summary output parent: {component}",
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError(
                    errno.ENOTDIR,
                    f"summary output parent is not a directory: {component}",
                )
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        result = directory_fd
        directory_fd = -1
        return result
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _require_missing_at(directory_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, f"summary output already exists: {name}")


def _open_exclusive_at(directory_fd: int, name: str) -> int:
    if Path(name).name != name:
        raise OSError(errno.EINVAL, "summary output name must be a basename")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise OSError(errno.EINVAL, "unsafe summary output file")
    return descriptor


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    return metadata.st_dev, metadata.st_ino


def _rollback_created_at(
    directory_fd: int, name: str, expected_identity: tuple[int, int]
) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == expected_identity
    ):
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass


def _require_commit_state(
    directory_fd: int,
    created_files: list[_CreatedOutput],
    output_directory: Path,
    directory_identity: tuple[int, int],
) -> None:
    for created in created_files:
        try:
            metadata = os.stat(
                created.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except OSError as error:
            raise SummaryError("summary output changed during commit") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != created.identity
            or metadata.st_size != created.size
        ):
            raise SummaryError("summary output changed during commit")

        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(
                created.name, flags, dir_fd=directory_fd
            )
        except OSError as error:
            raise SummaryError("summary output changed during commit") from error
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != created.identity
                or opened.st_size != created.size
                or _hash_descriptor(descriptor) != created.digest
            ):
                raise SummaryError("summary output changed during commit")
        finally:
            os.close(descriptor)
    _require_directory_binding(output_directory, directory_identity)


def _require_directory_binding(
    path: Path, expected_identity: tuple[int, int]
) -> None:
    try:
        descriptor = _open_directory_no_symlink(path, create=False)
    except OSError as error:
        raise SummaryError("summary output directory changed during commit") from error
    try:
        if _descriptor_identity(descriptor) != expected_identity:
            raise SummaryError("summary output directory changed during commit")
    finally:
        os.close(descriptor)


def _open_read_no_symlink(path: Path) -> int:
    try:
        directory_fd = _open_directory_no_symlink(path.parent, create=False)
    except OSError as error:
        raise SummaryError(f"cannot safely open summary input: {path.name}") from error
    try:
        metadata = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise SummaryError(
                f"summary input must be a regular file: {path.name}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            os.close(descriptor)
            raise SummaryError(f"summary input identity is unsafe: {path.name}")
        return descriptor
    except SummaryError:
        raise
    except OSError as error:
        raise SummaryError(f"cannot safely open summary input: {path.name}") from error
    finally:
        os.close(directory_fd)


def _copy_descriptor(descriptor: int, destination: Path) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        with destination.open("xb") as handle:
            while chunk := os.read(descriptor, 1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
        destination.chmod(stat.S_IRUSR)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise SummaryError("cannot create immutable summary input snapshot") from error
    return digest.hexdigest()


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise SummaryError("cannot verify held summary input") from error
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SummaryError(f"cannot hash summary input: {path.name}") from error
    return digest.hexdigest()
