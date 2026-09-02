"""Deterministic final technical-QC report generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import eddy_timing
from .qc import (
    FIGURE_IDS,
    FIGURE_FILENAMES,
    QCError,
    _DirectoryAnchor,
    _InputGuard,
    _OwnedWriteHandle,
    _commit_files,
    _load_outlier_map,
    _load_json,
    _load_numeric_table,
    _pin_output_directory,
    _private_temp,
    _validate_png,
)
from .resources import JHU_IMAGE_SHA256, JHU_XML_SHA256
from .state import StageContext, StageRecord, StageStateError
from .summary import CANONICAL_METRICS, COUNT_FIELDS


REPORT_JSON_KEYS = (
    "schema_version",
    "subject_id",
    "processing_status",
    "visual_review_status",
    "acquisition_assumptions",
    "input",
    "pre_denoise_motion_qc",
    "topup",
    "bet",
    "eddy",
    "models",
    "noddi",
    "atlas",
    "summary_mask",
    "global_values",
    "roi_summary",
    "stages",
    "software",
    "outputs",
    "limitations",
)
STAGE_METRIC_KEYS = (
    "denoise",
    "gibbs",
    "topup",
    "bet",
    "dti",
    "dki",
    "dki_direct",
    "noddi",
)
REPORT_STAGE_ORDER = (
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
)
_LIMITATIONS = (
    "Research use only.",
    "This report is not a clinical diagnosis.",
    "Acquisition parameters were user-supplied and were not independently "
    "verified from DICOM or scanner JSON.",
    "Visual inspection is still required; visual_review_status is NOT_REVIEWED.",
)


class ReportError(ValueError):
    """Raised when final-report inputs or outputs violate the contract."""


@dataclass(frozen=True)
class ReportContext:
    """Explicit inputs for one final subject report."""

    stage_context: StageContext
    output_directory: Path
    qc_manifest_json: Path
    input_audit_json: Path
    stripe_metrics_csv: Path
    stripe_decision_json: Path
    stripe_detail_files: tuple[Path, ...]
    stage_metrics_json: Mapping[str, Path]
    topup_field_hz: Path
    brain_mask: Path
    eddy_parameters: Path
    eddy_movement_rms: Path
    eddy_outlier_map: Path
    eddy_outlier_report: Path
    eddy_quad_json: Path
    eddy_timing_json: Path
    noddi_error_codes: Path
    summary_json: Path
    global_csv: Path
    roi_csv: Path
    atlas_provenance_json: Path
    stage_records: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage_context, StageContext):
            raise ReportError("stage_context must be a StageContext")
        for field in (
            "output_directory",
            "qc_manifest_json",
            "input_audit_json",
            "stripe_metrics_csv",
            "stripe_decision_json",
            "topup_field_hz",
            "brain_mask",
            "eddy_parameters",
            "eddy_movement_rms",
            "eddy_outlier_map",
            "eddy_outlier_report",
            "eddy_quad_json",
            "eddy_timing_json",
            "noddi_error_codes",
            "summary_json",
            "global_csv",
            "roi_csv",
            "atlas_provenance_json",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        object.__setattr__(
            self,
            "stripe_detail_files",
            tuple(Path(value) for value in self.stripe_detail_files),
        )
        if not self.stripe_detail_files:
            raise ReportError("stripe_detail_files must contain at least one detail sheet")
        object.__setattr__(
            self, "stage_records", tuple(Path(value) for value in self.stage_records)
        )
        try:
            metrics = dict(self.stage_metrics_json)
        except (TypeError, ValueError) as error:
            raise ReportError("stage_metrics_json must be a path mapping") from error
        if set(metrics) != set(STAGE_METRIC_KEYS):
            raise ReportError(
                f"stage_metrics_json must have exact keys {list(STAGE_METRIC_KEYS)}"
            )
        object.__setattr__(
            self,
            "stage_metrics_json",
            MappingProxyType(
                OrderedDict((key, Path(metrics[key])) for key in STAGE_METRIC_KEYS)
            ),
        )
        _validate_path_assignment(self)


@dataclass(frozen=True)
class ReportOutputs:
    """Paths to one transactional three-format report set."""

    pdf: Path
    markdown: Path
    run_summary_json: Path


@dataclass(frozen=True)
class _ReportData:
    payload: Mapping[str, object]
    figures: Mapping[str, Path]
    detail_files: tuple[Path, ...]
    snapshots: Mapping[Path, Path]


def write_final_report(context: ReportContext) -> ReportOutputs:
    """Write deterministic Markdown/JSON and a validated six-page PDF."""
    if not isinstance(context, ReportContext):
        raise ReportError("context must be a ReportContext")
    _validate_path_assignment(context)
    outputs = _output_paths(context)
    destination = _require_empty_output(context.output_directory)
    expected_figures = _expected_figure_paths(context)
    guarded = [*_context_inputs(context), *expected_figures.values()]
    try:
        try:
            with _InputGuard(guarded) as guard:
                figures = _validate_qc_manifest(
                    context,
                    guard,
                    expected_figures,
                )
                data = _collect_report_data(context, figures, guard)
                guard.verify()
                names = (
                    outputs.pdf.name,
                    outputs.markdown.name,
                    outputs.run_summary_json.name,
                )
                with _private_temp(
                    context.output_directory.parent,
                    names=names,
                ) as temporary:
                    json_path = (
                        temporary.path / outputs.run_summary_json.name
                    )
                    markdown_path = (
                        temporary.path / outputs.markdown.name
                    )
                    pdf_path = temporary.path / outputs.pdf.name
                    temporary.verify()
                    json_text = (
                        json.dumps(
                            _json_ready(data.payload),
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    _validate_serialized_report(json_text)
                    temporary.write_text(json_path.name, json_text)
                    temporary.verify()
                    markdown_text = _build_markdown(context, data)
                    _validate_serialized_report(markdown_text)
                    temporary.write_text(
                        markdown_path.name,
                        markdown_text,
                    )
                    temporary.verify()
                    with temporary.writer(pdf_path.name) as output:
                        _render_pdf(context, data, output)
                    temporary.verify()
                    _validate_pdf_bytes(
                        temporary.read_bytes(pdf_path.name)
                    )
                    _validate_markdown_links(
                        markdown_text,
                        context,
                        figures,
                    )
                    temporary.verify()
                    guard.verify()
                    _commit_files(
                        destination,
                        (
                            (pdf_path, pdf_path.name),
                            (markdown_path, markdown_path.name),
                            (json_path, json_path.name),
                        ),
                        require_empty=True,
                        verifier=guard.verify,
                        source_owner=temporary,
                    )
        except QCError as error:
            mapped = ReportError(str(error))
            for note in getattr(error, "__notes__", ()):
                mapped.add_note(note)
            raise mapped from error
    finally:
        destination.close()
    return outputs


def _validate_path_assignment(context: ReportContext) -> None:
    subject_root = context.stage_context.subject_root
    config = context.stage_context.config
    if subject_root.resolve(strict=False) != config.subject_output.resolve(strict=False):
        raise ReportError("StageContext subject_root must equal configured subject output")
    subject_paths = [
        context.qc_manifest_json,
        context.input_audit_json,
        context.stripe_metrics_csv,
        context.stripe_decision_json,
        *context.stripe_detail_files,
        *context.stage_metrics_json.values(),
        context.topup_field_hz,
        context.brain_mask,
        context.eddy_parameters,
        context.eddy_movement_rms,
        context.eddy_outlier_map,
        context.eddy_outlier_report,
        context.eddy_quad_json,
        context.eddy_timing_json,
        context.noddi_error_codes,
        context.summary_json,
        context.global_csv,
        context.roi_csv,
        *context.stage_records,
    ]
    all_inputs = [*subject_paths, context.atlas_provenance_json]
    identities: dict[tuple[int, int], Path] = {}
    for path in all_inputs:
        _require_regular(path)
        details = path.stat(follow_symlinks=False)
        identity = (int(details.st_dev), int(details.st_ino))
        if identity in identities:
            raise ReportError(f"report inputs must not be hard-link aliases: {path.name}")
        identities[identity] = path
    for path in subject_paths:
        _relative_to_subject(path, subject_root)
    _reject_path(context.output_directory)
    _relative_to_subject(
        context.output_directory, subject_root, require_exists=False
    )
    if os.path.lexists(context.output_directory):
        details = context.output_directory.lstat()
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise ReportError("report output directory must be a real directory")


def _context_inputs(context: ReportContext) -> list[Path]:
    return [
        context.qc_manifest_json,
        context.input_audit_json,
        context.stripe_metrics_csv,
        context.stripe_decision_json,
        *context.stripe_detail_files,
        *context.stage_metrics_json.values(),
        context.topup_field_hz,
        context.brain_mask,
        context.eddy_parameters,
        context.eddy_movement_rms,
        context.eddy_outlier_map,
        context.eddy_outlier_report,
        context.eddy_quad_json,
        context.eddy_timing_json,
        context.noddi_error_codes,
        context.summary_json,
        context.global_csv,
        context.roi_csv,
        context.atlas_provenance_json,
        *context.stage_records,
    ]


def _expected_figure_paths(context: ReportContext) -> Mapping[str, Path]:
    resolved: OrderedDict[str, Path] = OrderedDict()
    for figure_id in FIGURE_IDS:
        basename = FIGURE_FILENAMES[figure_id]
        if figure_id == "overview":
            basename = basename.replace(
                "${subject_id}", context.stage_context.config.subject_id
            )
        path = context.qc_manifest_json.parent / basename
        _relative_to_subject(path, context.stage_context.subject_root)
        resolved[figure_id] = path
    return MappingProxyType(resolved)


def _validate_qc_manifest(
    context: ReportContext,
    snapshots: _InputGuard,
    expected_figures: Mapping[str, Path],
) -> Mapping[str, Path]:
    payload = _read_json(snapshots.path(context.qc_manifest_json), "QC manifest")
    if tuple(payload) != (
        "schema_version",
        "subject_id",
        "visual_review_status",
        "figures",
        "figure_metadata",
        "pre_denoise_detail_directory",
    ):
        raise ReportError("QC manifest top-level schema/order is invalid")
    if (
        payload["schema_version"] != "1.0"
        or payload["subject_id"] != context.stage_context.config.subject_id
        or payload["visual_review_status"] != "NOT_REVIEWED"
    ):
        raise ReportError("QC manifest identity or review status is invalid")
    figures = payload["figures"]
    metadata = payload["figure_metadata"]
    if not isinstance(figures, dict) or not isinstance(metadata, dict):
        raise ReportError("QC manifest figures and metadata must be mappings")
    if tuple(figures) != FIGURE_IDS or tuple(metadata) != FIGURE_IDS:
        raise ReportError("QC manifest must contain the exact ordered 17 figure IDs")
    resolved: OrderedDict[str, Path] = OrderedDict()
    for figure_id in FIGURE_IDS:
        basename = figures[figure_id]
        figure_path = expected_figures[figure_id]
        expected = figure_path.name
        if basename != expected or Path(str(basename)).name != basename:
            raise ReportError("QC manifest contains an invalid figure basename")
        snapshot_path = snapshots.path(figure_path)
        width, height = _validate_png_report(snapshot_path)
        entry = metadata[figure_id]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sha256", "width", "height"}
            or entry["sha256"] != _sha256(snapshot_path)
            or entry["width"] != width
            or entry["height"] != height
        ):
            raise ReportError(f"QC manifest metadata mismatch for {figure_id}")
        resolved[figure_id] = figure_path
    detail_name = payload["pre_denoise_detail_directory"]
    if detail_name != context.stripe_metrics_csv.parent.name:
        raise ReportError("QC manifest pre-denoise detail pointer is inconsistent")
    return MappingProxyType(resolved)


def _collect_report_data(
    context: ReportContext,
    figures: Mapping[str, Path],
    snapshots: _InputGuard,
) -> _ReportData:
    frozen = snapshots.path
    subject = context.stage_context.config.subject_id
    for detail in context.stripe_detail_files:
        _validate_png_report(frozen(detail))
    audit = _read_json(frozen(context.input_audit_json), "input audit")
    _validate_input_audit(audit)
    stripe_rows = _read_stripe_rows(frozen(context.stripe_metrics_csv))
    if len(stripe_rows) != int(audit["pa_shape"][3]):
        raise ReportError("stripe CSV row count must equal audited PA volume count")
    stripe = _read_json(frozen(context.stripe_decision_json), "stripe decision")
    stripe_facts = _stripe_facts(stripe_rows, stripe, subject)
    metrics = {
        key: _read_json(frozen(path), f"{key} stage metrics")
        for key, path in context.stage_metrics_json.items()
    }
    for payload in metrics.values():
        if "subject_id" in payload and payload["subject_id"] != subject:
            raise ReportError("stage metrics subject_id does not match configuration")

    brain_image, brain = _load_3d(frozen(context.brain_mask), "brain mask")
    if (
        not np.equal(brain, np.rint(brain)).all()
        or not np.isin(brain, (0, 1)).all()
        or not np.any(brain == 1)
    ):
        raise ReportError("brain mask must be finite nonempty binary 0/1")
    field_image, field = _load_3d(frozen(context.topup_field_hz), "TOPUP field Hz")
    _same_grid(brain_image, field_image, "TOPUP field")
    field_values = field[brain == 1]
    topup = OrderedDict(
        (
            ("field_units", "Hz"),
            ("brain_mask_voxel_count", int(field_values.size)),
            ("field_median", float(np.median(field_values))),
            ("field_p01", float(np.percentile(field_values, 1))),
            ("field_p99", float(np.percentile(field_values, 99))),
            ("stage_metrics", _sanitize_metrics(metrics["topup"])),
        )
    )
    bet = _bet_facts(metrics["bet"], int(np.count_nonzero(brain)))

    pa_shape = audit.get("pa_shape")
    if not isinstance(pa_shape, list) or len(pa_shape) != 4:
        raise ReportError("input audit PA shape is invalid")
    volumes = int(pa_shape[3])
    eddy = _eddy_facts(context, volumes, snapshots)
    errors_image, errors = _load_3d(
        frozen(context.noddi_error_codes), "NODDI error codes"
    )
    _same_grid(brain_image, errors_image, "NODDI error codes")
    if not np.equal(errors, np.rint(errors)).all() or np.any(errors < 0):
        raise ReportError("NODDI error-code map must contain nonnegative integers")
    codes, counts = np.unique(errors[brain == 1].astype(np.int64), return_counts=True)
    histogram = [
        OrderedDict((("error_code", int(code)), ("voxel_count", int(count))))
        for code, count in zip(codes, counts, strict=True)
    ]
    noddi_metrics = metrics["noddi"]
    total_voxels = int(np.count_nonzero(brain))
    success_count = int(np.count_nonzero((brain == 1) & (errors == 0)))
    error_999_count = int(np.count_nonzero((brain == 1) & (errors == 999)))
    other_error_count = total_voxels - success_count - error_999_count
    for key, value in (
        ("success_count", success_count),
        ("error_999_count", error_999_count),
        ("other_error_count", other_error_count),
    ):
        if noddi_metrics.get(key) != value:
            raise ReportError(f"NODDI metrics {key} disagrees with the error-code map")
    noddi = OrderedDict(
        (
            ("error_code_scope", "cleaned_brain_mask"),
            ("error_code_histogram", histogram),
            ("stage_metrics", _sanitize_metrics(noddi_metrics)),
        )
    )

    global_row = _read_global_csv(frozen(context.global_csv), subject)
    roi = _read_roi_csv(frozen(context.roi_csv), subject, global_row)
    summary = _read_json(frozen(context.summary_json), "subject summary")
    _validate_summary(summary, subject, global_row, roi, histogram)
    provenance = _read_json(
        frozen(context.atlas_provenance_json), "atlas provenance"
    )
    atlas = _atlas_facts(provenance, summary)
    stages, stage_software = _stage_facts(context, subject, snapshots)

    model_facts = OrderedDict(
        (
            (key, _model_facts(metrics[key]))
            for key in ("dti", "dki", "dki_direct")
        )
    )
    model_facts["noddi"] = _model_facts(metrics["noddi"])

    acquisition = context.stage_context.config.acquisition
    assumptions = OrderedDict(
        (
            ("pa_phase_encoding_vector", list(acquisition.pa_vector)),
            ("ap_phase_encoding_vector", list(acquisition.ap_vector)),
            ("total_readout_time_seconds", acquisition.total_readout_time),
            ("source", "user-supplied"),
            ("independently_verified_from_dicom_or_json", False),
            (
                "statement",
                "These values were user-supplied and were not inferred or "
                "independently verified from DICOM or scanner JSON.",
            ),
        )
    )
    input_facts = OrderedDict(
        (
            ("pa_shape", audit["pa_shape"]),
            ("ap_shape", audit["ap_shape"]),
            ("pa_voxel_size_mm", audit["pa_spatial_zooms"]),
            ("ap_voxel_size_mm", audit["ap_spatial_zooms"]),
            ("shell_distribution", audit["shell_counts"]),
        )
    )
    summary_mask = OrderedDict(
        (field, int(global_row[field])) for field in COUNT_FIELDS
    )
    global_values = OrderedDict(
        (metric, float(global_row[metric])) for metric in CANONICAL_METRICS
    )
    outputs = _report_links(context, figures)
    software = OrderedDict(
        (key, _sanitize_software(value))
        for key, value in sorted(
            {**dict(context.stage_context.software), **stage_software}.items()
        )
    )
    payload = OrderedDict(
        (
            ("schema_version", "1.0"),
            ("subject_id", subject),
            ("processing_status", "COMPLETED"),
            ("visual_review_status", "NOT_REVIEWED"),
            ("acquisition_assumptions", assumptions),
            ("input", input_facts),
            ("pre_denoise_motion_qc", stripe_facts),
            ("topup", topup),
            ("bet", bet),
            ("eddy", eddy),
            ("models", model_facts),
            ("noddi", noddi),
            ("atlas", atlas),
            ("summary_mask", summary_mask),
            ("global_values", global_values),
            (
                "roi_summary",
                OrderedDict(
                    (
                        ("row_count", len(roi)),
                        (
                            "voxel_count_total",
                            int(sum(int(row["voxel_count"]) for row in roi)),
                        ),
                        ("true_label_ids", list(range(1, 49))),
                    )
                ),
            ),
            ("stages", stages),
            ("software", software),
            ("outputs", outputs),
            ("limitations", list(_LIMITATIONS)),
        )
    )
    if tuple(payload) != REPORT_JSON_KEYS:
        raise ReportError("internal report JSON schema order changed")
    _finite_recursive(payload, "report payload")
    return _ReportData(
        MappingProxyType(payload),
        figures,
        tuple(context.stripe_detail_files),
        snapshots.paths,
    )


def _validate_input_audit(payload: Mapping[str, object]) -> None:
    required = {
        "pa_shape",
        "ap_shape",
        "pa_spatial_zooms",
        "ap_spatial_zooms",
        "shell_counts",
    }
    if not required.issubset(payload):
        raise ReportError("input audit is missing required shape/shell fields")
    for key in ("pa_shape", "ap_shape", "pa_spatial_zooms", "ap_spatial_zooms"):
        values = payload[key]
        if not isinstance(values, list) or not values:
            raise ReportError(f"input audit {key} is invalid")
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ReportError(f"input audit {key} must be finite")
    pa_shape = payload["pa_shape"]
    ap_shape = payload["ap_shape"]
    if (
        len(pa_shape) != 4
        or len(ap_shape) not in (3, 4)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (*pa_shape, *ap_shape)
        )
    ):
        raise ReportError("input audit PA/AP shapes are invalid")
    for key in ("pa_spatial_zooms", "ap_spatial_zooms"):
        values = payload[key]
        if len(values) != 3 or any(float(value) <= 0 for value in values):
            raise ReportError(f"input audit {key} must have three positive values")
    shells = payload["shell_counts"]
    if not isinstance(shells, dict) or not shells:
        raise ReportError("input audit shell distribution is invalid")
    if any(
        not str(key).lstrip("-").isdigit()
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for key, value in shells.items()
    ):
        raise ReportError("input audit shell counts are invalid")
    if sum(int(value) for value in shells.values()) != int(pa_shape[3]):
        raise ReportError("input audit shell counts must sum to PA volume count")


def _read_stripe_rows(path: Path) -> list[dict[str, object]]:
    required = {
        "volume_index_zero_based",
        "volume_number_one_based",
        "c_si",
        "classification",
    }
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ReportError("stripe metrics CSV lacks required columns")
            raw = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReportError("cannot read stripe metrics CSV") from error
    rows: list[dict[str, object]] = []
    for index, row in enumerate(raw):
        try:
            zero = int(row["volume_index_zero_based"])
            one = int(row["volume_number_one_based"])
            csi = float(row["c_si"])
        except (TypeError, ValueError) as error:
            raise ReportError("stripe CSV has malformed values") from error
        classification = row["classification"]
        expected = "high" if csi > 1.25 else "ambiguous" if csi >= 1.15 else "normal"
        if zero != index or one != index + 1 or not math.isfinite(csi) or classification != expected:
            raise ReportError("stripe CSV indices/classification are inconsistent")
        rows.append(
            {"index": zero, "number": one, "csi": csi, "classification": classification}
        )
    if not rows:
        raise ReportError("stripe CSV must contain at least one DWI row")
    return rows


def _stripe_facts(
    rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    subject: str,
) -> Mapping[str, object]:
    if decision.get("subject_id") != subject:
        raise ReportError("stripe decision subject_id is inconsistent")
    high = [int(row["number"]) for row in rows if row["classification"] == "high"]
    ambiguous = [
        int(row["number"]) for row in rows if row["classification"] == "ambiguous"
    ]
    maximum_index = int(np.argmax([float(row["csi"]) for row in rows]))
    maximum = float(rows[maximum_index]["csi"])
    if decision.get("thresholds") != {
        "ambiguous_min_inclusive": 1.15,
        "high_min_exclusive": 1.25,
        "exclude_high_volume_count": 5,
    }:
        raise ReportError("stripe decision thresholds are invalid")
    if decision.get("flagged_volume_numbers_one_based") != {
        "high": high,
        "ambiguous": ambiguous,
    }:
        raise ReportError("stripe decision flagged volumes disagree with CSV")
    if decision.get("volume_counts") != {
        "total": len(rows),
        "normal": len(rows) - len(high) - len(ambiguous),
        "ambiguous": len(ambiguous),
        "high": len(high),
    }:
        raise ReportError("stripe decision counts disagree with CSV")
    recorded_max = decision.get("maximum_csi")
    if (
        not isinstance(recorded_max, dict)
        or recorded_max.get("volume_number_one_based") != maximum_index + 1
        or not math.isclose(float(recorded_max.get("value", math.nan)), maximum)
    ):
        raise ReportError("stripe decision maximum cSI disagrees with CSV")
    return OrderedDict(
        (
            ("decision", decision.get("decision")),
            ("high_volume_count", len(high)),
            ("ambiguous_volume_count", len(ambiguous)),
            ("maximum_csi", maximum),
            ("maximum_csi_volume_one_based", maximum_index + 1),
            ("high_volumes_one_based", high),
            ("ambiguous_volumes_one_based", ambiguous),
        )
    )


def _bet_facts(payload: Mapping[str, object], mask_count: int) -> Mapping[str, object]:
    keys = (
        "component_count",
        "original_voxel_count",
        "largest_voxel_count",
        "removed_voxel_count",
        "largest_tie",
    )
    if not all(key in payload for key in keys):
        raise ReportError("BET metrics are missing component cleanup facts")
    if payload["largest_voxel_count"] != mask_count:
        raise ReportError("BET largest component count disagrees with cleaned mask")
    for key in keys[:4]:
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            raise ReportError("BET voxel/component metrics must be integers")
    if not isinstance(payload["largest_tie"], bool):
        raise ReportError("BET largest_tie must be boolean")
    if (
        payload["component_count"] < 1
        or payload["largest_voxel_count"] < 1
        or payload["original_voxel_count"] < payload["largest_voxel_count"]
        or payload["removed_voxel_count"]
        != payload["original_voxel_count"] - payload["largest_voxel_count"]
    ):
        raise ReportError("BET component/voxel metrics are internally inconsistent")
    return OrderedDict(
        (
            ("component_count", payload["component_count"]),
            ("original_voxel_count", payload["original_voxel_count"]),
            ("largest_voxel_count", payload["largest_voxel_count"]),
            ("removed_voxel_count", payload["removed_voxel_count"]),
            ("largest_tie", payload["largest_tie"]),
            (
                "tie_warning",
                "Largest-component tie resolved deterministically by first label."
                if payload["largest_tie"]
                else None,
            ),
        )
    )


def _eddy_facts(
    context: ReportContext, volumes: int, snapshots: _InputGuard
) -> Mapping[str, object]:
    parameters = _load_table(
        snapshots.path(context.eddy_parameters), "EDDY parameters"
    )
    movement = _load_table(
        snapshots.path(context.eddy_movement_rms), "EDDY movement RMS"
    )
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    if movement.ndim == 1:
        movement = movement[None, :]
    if parameters.shape[0] != volumes or parameters.shape[1] < 6:
        raise ReportError("EDDY parameters have the wrong row/column count")
    if movement.shape[0] != volumes or movement.shape[1] < 2:
        raise ReportError("EDDY RMS has the wrong row/column count")
    if np.any(movement[:, :2] < 0):
        raise ReportError("EDDY RMS values must be nonnegative")
    try:
        outliers = _load_outlier_map(
            snapshots.path(context.eddy_outlier_map), volumes
        )
    except QCError as error:
        raise ReportError(str(error)) from error
    outlier_counts = np.sum(outliers, axis=1).astype(int)
    observed = int(np.sum(outlier_counts))
    affected = int(np.count_nonzero(outlier_counts))
    reported = _reported_outlier_count(
        snapshots.path(context.eddy_outlier_report)
    )
    if reported != observed:
        raise ReportError("EDDY reported and observed outlier-slice totals disagree")
    quad = _sanitize_eddy_quad(
        _read_json(snapshots.path(context.eddy_quad_json), "EDDY QUAD")
    )
    try:
        timing = eddy_timing.read_eddy_timing(
            snapshots.path(context.eddy_timing_json)
        )
    except eddy_timing.EddyTimingError as error:
        raise ReportError(str(error)) from error
    return OrderedDict(
        (
            (
                "translation_max_abs_mm",
                float(np.max(np.abs(parameters[:, :3]))),
            ),
            (
                "rotation_max_abs_degrees",
                float(np.max(np.abs(np.rad2deg(parameters[:, 3:6])))),
            ),
            ("absolute_rms_mean_mm", float(np.mean(movement[:, 0]))),
            ("absolute_rms_max_mm", float(np.max(movement[:, 0]))),
            ("relative_rms_mean_mm", float(np.mean(movement[:, 1]))),
            ("relative_rms_max_mm", float(np.max(movement[:, 1]))),
            ("reported_outlier_slice_count", reported),
            ("observed_outlier_slice_count", observed),
            ("affected_volume_count", affected),
            ("maximum_slices_in_one_volume", int(np.max(outlier_counts))),
            ("eddy_quad_selected_numeric_metrics", quad),
            (
                "runtime_seconds",
                OrderedDict(
                    (
                        (
                            "eddy_command_including_cnr_and_residuals",
                            timing.eddy_command_seconds,
                        ),
                        ("eddy_quad", timing.eddy_quad_seconds),
                        ("stage_action_total", timing.stage_action_seconds),
                    )
                ),
            ),
        )
    )


def _reported_outlier_count(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReportError("cannot read EDDY outlier report") from error
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0
    pattern = re.compile(
        r"Slice\s+\d+\s+in\s+scan\s+\d+\s+is\s+an\s+outlier\b.*",
        flags=re.IGNORECASE,
    )
    if not all(pattern.fullmatch(line) for line in lines):
        raise ReportError("unsupported EDDY outlier report format")
    return len(lines)


def _sanitize_eddy_quad(payload: Mapping[str, object]) -> Mapping[str, float]:
    selected: OrderedDict[str, float] = OrderedDict()
    accepted = re.compile(
        r"(?i)(motion|mot_|outlier|cnr|snr|rms|displacement|qc_)"
    )

    def visit(value: object, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                child = value[key]
                label = f"{prefix}.{key}" if prefix else key
                visit(child, label)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and accepted.search(prefix)
        ):
            number = float(value)
            if math.isfinite(number):
                selected[prefix] = number

    visit(payload)
    return MappingProxyType(selected)


def _model_facts(payload: Mapping[str, object]) -> Mapping[str, object]:
    selected: OrderedDict[str, object] = OrderedDict()
    for key in (
        "model",
        "model_name",
        "selection",
        "dti_max_b",
        "selected_volume_count",
        "selected_b0_count",
        "shells",
        "worker_count",
        "success_count",
        "error_999_count",
        "other_error_count",
        "nonfinite_replaced",
    ):
        if key in payload:
            selected[key] = _sanitize_metrics(payload[key])
    warnings: list[str] = []
    replacements = payload.get("nonfinite_replaced")
    if isinstance(replacements, dict):
        total = 0
        for value in replacements.values():
            if isinstance(value, dict):
                total += sum(
                    int(count)
                    for count in value.values()
                    if isinstance(count, int) and not isinstance(count, bool)
                )
        if total:
            warnings.append(f"{total} non-finite map values were replaced with zero.")
    selected["warnings"] = warnings
    return MappingProxyType(selected)


def _read_global_csv(path: Path, subject: str) -> Mapping[str, object]:
    expected = ("subject_id", *COUNT_FIELDS, *CANONICAL_METRICS)
    rows = _read_csv(path)
    if len(rows) != 1 or tuple(rows[0]) != expected:
        raise ReportError("global CSV must have one row and exact canonical columns")
    row = rows[0]
    if row["subject_id"] != subject:
        raise ReportError("global CSV subject_id is inconsistent")
    parsed: OrderedDict[str, object] = OrderedDict((("subject_id", subject),))
    for field in COUNT_FIELDS:
        try:
            value = int(row[field])
        except ValueError as error:
            raise ReportError("global count fields must be integers") from error
        if value < 0:
            raise ReportError("global count fields must be nonnegative")
        parsed[field] = value
    counts = [int(parsed[field]) for field in COUNT_FIELDS]
    if any(later > earlier for earlier, later in zip(counts, counts[1:])):
        raise ReportError("global summary-mask counts must be monotonic")
    for metric in CANONICAL_METRICS:
        try:
            value = float(row[metric])
        except ValueError as error:
            raise ReportError("global metric fields must be numeric") from error
        if not math.isfinite(value):
            raise ReportError("global metric fields must be finite")
        parsed[metric] = value
    return MappingProxyType(parsed)


def _read_roi_csv(
    path: Path, subject: str, global_row: Mapping[str, object]
) -> tuple[Mapping[str, object], ...]:
    expected = ("subject_id", "label_id", "label_name", "voxel_count", *CANONICAL_METRICS)
    rows = _read_csv(path)
    if len(rows) != 48 or any(tuple(row) != expected for row in rows):
        raise ReportError("ROI CSV must contain exactly 48 canonical rows")
    parsed: list[Mapping[str, object]] = []
    for expected_id, row in enumerate(rows, 1):
        try:
            label_id = int(row["label_id"])
            voxel_count = int(row["voxel_count"])
            values = [float(row[key]) for key in CANONICAL_METRICS]
        except ValueError as error:
            raise ReportError("ROI CSV contains malformed numeric values") from error
        if (
            row["subject_id"] != subject
            or label_id != expected_id
            or not row["label_name"]
            or voxel_count < 1
            or not np.isfinite(values).all()
        ):
            raise ReportError("ROI CSV true labels, names, counts, or metrics are invalid")
        parsed.append(
            MappingProxyType(
                {
                    "label_id": label_id,
                    "label_name": row["label_name"],
                    "voxel_count": voxel_count,
                }
            )
        )
    total = sum(int(row["voxel_count"]) for row in parsed)
    if total != global_row["common_mask_voxel_count"]:
        raise ReportError("48 ROI voxel total must equal common-mask voxel count")
    return tuple(parsed)


def _validate_summary(
    payload: Mapping[str, object],
    subject: str,
    global_row: Mapping[str, object],
    roi: Sequence[Mapping[str, object]],
    histogram: Sequence[Mapping[str, object]],
) -> None:
    if payload.get("subject_id") != subject:
        raise ReportError("summary subject_id is inconsistent")
    common = payload.get("common_mask")
    if not isinstance(common, dict) or common.get("counts") != {
        field: global_row[field] for field in COUNT_FIELDS
    }:
        raise ReportError("summary-mask counts disagree with global CSV")
    counts = payload.get("roi_voxel_counts")
    expected = {str(row["label_id"]): row["voxel_count"] for row in roi}
    if counts != expected:
        raise ReportError("summary ROI voxel counts disagree with ROI CSV")
    noddi_histogram = payload.get("noddi_error_code_histogram")
    if (
        not isinstance(noddi_histogram, dict)
        or noddi_histogram.get("scope") != "cleaned_brain_mask"
        or noddi_histogram.get("bins") != [dict(value) for value in histogram]
    ):
        raise ReportError("summary NODDI error histogram disagrees with the primary map")


def _atlas_facts(
    provenance: Mapping[str, object], summary: Mapping[str, object]
) -> Mapping[str, object]:
    expected_files = {
        "JHU-ICBM-labels-2mm.nii.gz": {"sha256": JHU_IMAGE_SHA256},
        "JHU-labels.xml": {"sha256": JHU_XML_SHA256},
    }
    if provenance.get("source") != {"component": "data_atlases", "tag": "fsl-5_0_4"}:
        raise ReportError("atlas provenance source/version is invalid")
    if provenance.get("files") != expected_files:
        raise ReportError("atlas provenance hashes are invalid")
    if provenance.get("nonzero_labels") != list(range(1, 49)):
        raise ReportError("atlas provenance must record labels 1 through 48")
    atlas_summary = summary.get("atlas")
    if (
        not isinstance(atlas_summary, dict)
        or atlas_summary.get("nonzero_labels") != list(range(1, 49))
        or atlas_summary.get("source") != provenance["source"]
        or atlas_summary.get("resource_files") != expected_files
        or not isinstance(atlas_summary.get("warped_atlas_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(atlas_summary.get("warped_atlas_sha256"))
        )
        is None
    ):
        raise ReportError("summary atlas labels are invalid")
    return OrderedDict(
        (
            ("name", "JHU ICBM labels 2 mm"),
            ("version", "FSL data_atlases fsl-5_0_4"),
            ("resource_sha256", expected_files),
            ("warped_atlas_sha256", atlas_summary.get("warped_atlas_sha256")),
            ("nonzero_labels", list(range(1, 49))),
            ("interpolation", "nearest-neighbour only"),
        )
    )


def _stage_facts(
    context: ReportContext, subject: str, snapshots: _InputGuard
) -> tuple[list[Mapping[str, object]], dict[str, str]]:
    if len(context.stage_records) != len(REPORT_STAGE_ORDER):
        raise ReportError("stage records must include every completed stage through QC")
    stages: list[Mapping[str, object]] = []
    software: dict[str, str] = {}
    names: list[str] = []
    for path in context.stage_records:
        payload = _read_json(
            snapshots.path(path), "stage completion record"
        )
        try:
            record = StageRecord.from_dict(payload)
        except StageStateError as error:
            raise ReportError("stage completion record is invalid") from error
        if record.subject_id != subject:
            raise ReportError("stage completion record subject_id is inconsistent")
        if "report" in record.stage.lower():
            raise ReportError("report stage itself must not appear in input stage records")
        if record.stage in names:
            raise ReportError("stage completion records must not contain duplicates")
        names.append(record.stage)
        outputs: list[dict[str, object]] = []
        for entry in record.outputs:
            relative = str(entry["relative_path"])
            _safe_record_relative(relative)
            outputs.append(
                {
                    "relative_path": relative,
                    "sha256": entry["sha256"],
                    "size": entry["size"],
                }
            )
        stages.append(
            MappingProxyType(
                OrderedDict(
                    (
                        ("stage", record.stage),
                        ("started_utc", record.started_utc),
                        ("completed_utc", record.completed_utc),
                        ("stage_signature", record.stage_signature),
                        ("config_sha256", record.config_sha256),
                        (
                            "input_hashes",
                            [entry["sha256"] for entry in record.inputs],
                        ),
                        ("outputs", outputs),
                    )
                )
            )
        )
        software.update(
            (key, _sanitize_software(value))
            for key, value in record.software.items()
        )
    if tuple(names) != REPORT_STAGE_ORDER:
        raise ReportError("stage completion records are missing, extra, or out of order")
    return stages, software


def _report_links(
    context: ReportContext, figures: Mapping[str, Path]
) -> Mapping[str, object]:
    links: OrderedDict[str, object] = OrderedDict()
    links["qc_figures"] = OrderedDict(
        (figure_id, _safe_link(context, path)) for figure_id, path in figures.items()
    )
    links["stripe_detail_sheets"] = [
        _safe_link(context, path) for path in context.stripe_detail_files
    ]
    links["global_csv"] = _safe_link(context, context.global_csv)
    links["roi_csv"] = _safe_link(context, context.roi_csv)
    links["summary_json"] = _safe_link(context, context.summary_json)
    outputs = _output_paths(context)
    links["report_pdf"] = outputs.pdf.name
    links["report_markdown"] = outputs.markdown.name
    links["run_summary_json"] = outputs.run_summary_json.name
    return MappingProxyType(links)


def _build_markdown(context: ReportContext, data: _ReportData) -> str:
    payload = data.payload
    subject = payload["subject_id"]
    global_values = payload["global_values"]
    assert isinstance(global_values, Mapping)
    outputs = payload["outputs"]
    assert isinstance(outputs, Mapping)
    qc_links = outputs["qc_figures"]
    assert isinstance(qc_links, Mapping)
    assumptions = payload["acquisition_assumptions"]
    input_facts = payload["input"]
    stripe = payload["pre_denoise_motion_qc"]
    topup = payload["topup"]
    bet = payload["bet"]
    eddy = payload["eddy"]
    models = payload["models"]
    noddi = payload["noddi"]
    atlas = payload["atlas"]
    summary_mask = payload["summary_mask"]
    for value in (
        assumptions,
        input_facts,
        stripe,
        topup,
        bet,
        eddy,
        models,
        noddi,
        atlas,
        summary_mask,
    ):
        assert isinstance(value, Mapping)
    runtime = eddy["runtime_seconds"]
    assert isinstance(runtime, Mapping)
    command_runtime = float(
        runtime["eddy_command_including_cnr_and_residuals"]
    )

    lines = [
        f"# {subject} diffusion MRI analysis report",
        "",
        "Processing status: COMPLETED  ",
        "visual_review_status: NOT_REVIEWED",
        "",
        "## Acquisition assumptions",
        "",
    ]

    acquisition_rows: list[tuple[object, ...]] = []
    for direction, vector in (
        ("PA phase-encoding vector", assumptions["pa_phase_encoding_vector"]),
        ("AP phase-encoding vector", assumptions["ap_phase_encoding_vector"]),
    ):
        for component, value in zip(("x", "y", "z"), vector, strict=True):
            acquisition_rows.append((direction, component, value))
    acquisition_rows.extend(
        (
            (
                "TotalReadoutTime",
                "seconds",
                assumptions["total_readout_time_seconds"],
            ),
            ("Acquisition-parameter source", "value", assumptions["source"]),
            (
                "Independently verified from DICOM or scanner JSON",
                "value",
                assumptions["independently_verified_from_dicom_or_json"],
            ),
        )
    )
    _append_markdown_table(
        lines,
        ("Parameter", "Component or unit", "Value"),
        acquisition_rows,
    )
    lines.extend(
        [
            "",
            _markdown_cell(assumptions["statement"]),
            "",
            "## Input and shell distribution",
            "",
        ]
    )

    geometry_rows: list[tuple[object, ...]] = []
    for series, shape, voxel_size in (
        ("PA", input_facts["pa_shape"], input_facts["pa_voxel_size_mm"]),
        ("AP", input_facts["ap_shape"], input_facts["ap_voxel_size_mm"]),
    ):
        shape_components = ("x", "y", "z", "volumes")
        for component, value in zip(
            shape_components[: len(shape)], shape, strict=True
        ):
            geometry_rows.append((series, "shape", component, value))
        for component, value in zip(("x", "y", "z"), voxel_size, strict=True):
            geometry_rows.append((series, "voxel size (mm)", component, value))
    _append_markdown_table(
        lines,
        ("Series", "Field", "Component", "Value"),
        geometry_rows,
    )
    lines.extend(["", "### Nominal shell counts", ""])
    shell_distribution = input_facts["shell_distribution"]
    assert isinstance(shell_distribution, Mapping)
    _append_markdown_table(
        lines,
        ("Nominal shell", "Volume count"),
        [
            (shell, count)
            for shell, count in sorted(
                shell_distribution.items(), key=lambda item: int(item[0])
            )
        ],
    )
    lines.extend(
        [
            "",
            "## Pre-denoise motion/stripe QC",
            "",
            f"- Automatic decision: `{_markdown_cell(stripe['decision'])}`",
            f"- High volume count: `{_markdown_cell(stripe['high_volume_count'])}`",
            (
                "- Ambiguous volume count: "
                f"`{_markdown_cell(stripe['ambiguous_volume_count'])}`"
            ),
            f"- Maximum cSI: `{float(stripe['maximum_csi']):.6g}`",
            (
                "- Maximum cSI volume (one-based): "
                f"`{_markdown_cell(stripe['maximum_csi_volume_one_based'])}`"
            ),
            "",
            "### Flagged volumes",
            "",
        ]
    )
    flagged_rows: list[tuple[object, ...]] = []
    for classification, values in (
        ("High", stripe["high_volumes_one_based"]),
        ("Ambiguous", stripe["ambiguous_volumes_one_based"]),
    ):
        if values:
            flagged_rows.extend(
                (classification, value) for value in values
            )
        else:
            flagged_rows.append((classification, "none"))
    _append_markdown_table(
        lines,
        ("Classification", "One-based volume"),
        flagged_rows,
    )
    lines.extend(["", f"- [Stripe overview]({qc_links['stripe']})"])
    for index, link in enumerate(outputs["stripe_detail_sheets"], 1):
        lines.append(f"- [Stripe detail sheet {index}]({link})")

    lines.extend(
        [
            "",
            "## TOPUP, BET, and EDDY technical QC",
            "",
            "### TOPUP",
            "",
        ]
    )
    _append_markdown_table(
        lines,
        ("Metric", "Value", "Unit"),
        (
            ("Brain-mask voxel count", topup["brain_mask_voxel_count"], "voxels"),
            ("Field median", f"{float(topup['field_median']):.6g}", "Hz"),
            ("Field p01", f"{float(topup['field_p01']):.6g}", "Hz"),
            ("Field p99", f"{float(topup['field_p99']):.6g}", "Hz"),
        ),
    )
    lines.extend(["", "### BET", ""])
    _append_markdown_table(
        lines,
        ("Metric", "Value"),
        (
            ("Component count", bet["component_count"]),
            ("Original voxel count", bet["original_voxel_count"]),
            ("Largest-component voxel count", bet["largest_voxel_count"]),
            ("Removed voxel count", bet["removed_voxel_count"]),
            ("Largest-component tie", bet["largest_tie"]),
            ("Tie warning", bet["tie_warning"]),
        ),
    )
    lines.extend(
        [
            "",
            "### EDDY",
            "",
            (
                "- Maximum outlier slices in one volume: "
                f"`{_markdown_cell(eddy['maximum_slices_in_one_volume'])}`"
            ),
            "",
        ]
    )
    _append_markdown_table(
        lines,
        ("Metric", "Unit", "Value"),
        (
            (
                "Maximum absolute translation",
                "mm",
                f"{float(eddy['translation_max_abs_mm']):.6g}",
            ),
            (
                "Maximum absolute rotation",
                "degrees",
                f"{float(eddy['rotation_max_abs_degrees']):.6g}",
            ),
            (
                "Absolute RMS mean",
                "mm",
                f"{float(eddy['absolute_rms_mean_mm']):.6g}",
            ),
            (
                "Absolute RMS maximum",
                "mm",
                f"{float(eddy['absolute_rms_max_mm']):.6g}",
            ),
            (
                "Relative RMS mean",
                "mm",
                f"{float(eddy['relative_rms_mean_mm']):.6g}",
            ),
            (
                "Relative RMS maximum",
                "mm",
                f"{float(eddy['relative_rms_max_mm']):.6g}",
            ),
            (
                "Reported outlier-slice count",
                "slices",
                eddy["reported_outlier_slice_count"],
            ),
            (
                "Observed outlier-slice count",
                "slices",
                eddy["observed_outlier_slice_count"],
            ),
            (
                "Affected volume count",
                "volumes",
                eddy["affected_volume_count"],
            ),
            (
                "Maximum slices in one volume",
                "slices",
                eddy["maximum_slices_in_one_volume"],
            ),
            (
                "EDDY command (includes CNR maps and residuals)",
                "seconds",
                f"{command_runtime:.12g}",
            ),
            (
                "EDDY QUAD",
                "seconds",
                f"{float(runtime['eddy_quad']):.12g}",
            ),
            (
                "05_eddy stage action total",
                "seconds",
                f"{float(runtime['stage_action_total']):.12g}",
            ),
        ),
    )
    lines.extend(["", "### Selected numeric EDDY QUAD metrics", ""])
    quad = eddy["eddy_quad_selected_numeric_metrics"]
    assert isinstance(quad, Mapping)
    quad_rows = (
        [(key, value) for key, value in quad.items()]
        if quad
        else [("Selected numeric metrics", "none")]
    )
    _append_markdown_table(lines, ("Metric", "Value"), quad_rows)
    lines.extend(
        [
            "",
            "## Models and warnings",
            "",
        ]
    )
    model_rows: list[tuple[object, ...]] = []
    for name, facts in models.items():
        assert isinstance(facts, Mapping)
        for field, value in _flatten_markdown_fields(facts):
            model_rows.append((name, field, value))
    _append_markdown_table(
        lines,
        ("Model", "Field", "Value"),
        model_rows,
    )
    lines.extend(
        [
            "",
            "## NODDI error-code QC",
            "",
            f"- Scope: `{_markdown_cell(noddi['error_code_scope'])}`",
            "",
        ]
    )
    histogram = noddi["error_code_histogram"]
    assert isinstance(histogram, Sequence)
    _append_markdown_table(
        lines,
        ("Error code", "Voxel count"),
        [
            (entry["error_code"], entry["voxel_count"])
            for entry in histogram
        ],
    )
    lines.extend(
        [
            "",
            "## Global values",
            "",
            "| Metric | Global value |",
            "|---|---:|",
        ]
    )
    for metric, value in global_values.items():
        lines.append(
            f"| {_markdown_cell(metric)} | {_markdown_cell(float(value))} |"
        )
    lines.extend(
        [
            "",
            "## JHU 48-ROI summary",
            "",
            "### Atlas facts",
            "",
        ]
    )
    roi_summary = payload["roi_summary"]
    assert isinstance(roi_summary, Mapping)
    _append_markdown_table(
        lines,
        ("Field", "Value"),
        (
            ("Name", atlas["name"]),
            ("Version", atlas["version"]),
            ("Warped atlas SHA-256", atlas["warped_atlas_sha256"]),
            ("Nonzero labels", "1–48"),
            ("Interpolation", atlas["interpolation"]),
            ("ROI rows", roi_summary["row_count"]),
            ("ROI voxel total", roi_summary["voxel_count_total"]),
        ),
    )
    lines.extend(["", "### Atlas resource hashes", ""])
    resource_hashes = atlas["resource_sha256"]
    assert isinstance(resource_hashes, Mapping)
    resource_rows: list[tuple[object, ...]] = []
    for resource, details in resource_hashes.items():
        assert isinstance(details, Mapping)
        for field, value in _flatten_markdown_fields(details):
            resource_rows.append((resource, field, value))
    _append_markdown_table(
        lines,
        ("Resource", "Field", "Value"),
        resource_rows,
    )
    lines.extend(["", "### Summary-mask counts", ""])
    _append_markdown_table(
        lines,
        ("Mask stage", "Voxel count"),
        [(field, count) for field, count in summary_mask.items()],
    )
    lines.extend(
        [
            "",
            f"- [48-ROI table]({outputs['roi_csv']})",
            f"- [Global table]({outputs['global_csv']})",
            f"- [Summary JSON]({outputs['summary_json']})",
            "",
            "## Complete technical QC figures",
            "",
        ]
    )
    for figure_id, link in qc_links.items():
        lines.append(f"- [{figure_id.replace('_', ' ')}]({link})")
    lines.extend(
        [
            "",
            "## Ordered stage provenance",
            "",
            "| Stage | Started UTC | Completed UTC | Signature |",
            "|---|---|---|---|",
        ]
    )
    for stage in payload["stages"]:
        lines.append(
            f"| {_markdown_cell(stage['stage'])} | "
            f"{_markdown_cell(stage['started_utc'])} | "
            f"{_markdown_cell(stage['completed_utc'])} | "
            f"`{_markdown_cell(stage['stage_signature'])}` |"
        )
    lines.extend(
        [
            "",
            "## Stage hash and output detail",
            "",
            "| Stage | Config SHA-256 | Input SHA-256 values | Output | Output SHA-256 | Bytes |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for stage in payload["stages"]:
        input_hashes = "<br>".join(
            f"`{_markdown_cell(value)}`" for value in stage["input_hashes"]
        )
        stage_outputs = stage["outputs"] or (
            {"relative_path": "—", "sha256": "—", "size": 0},
        )
        for output in stage_outputs:
            lines.append(
                f"| {_markdown_cell(stage['stage'])} | "
                f"`{_markdown_cell(stage['config_sha256'])}` | "
                f"{input_hashes} | {_markdown_cell(output['relative_path'])} | "
                f"`{_markdown_cell(output['sha256'])}` | "
                f"{_markdown_cell(output['size'])} |"
            )
    lines.extend(
        [
            "",
            "## Software versions",
            "",
            "| Component | Version |",
            "|---|---|",
        ]
    )
    for component, version in payload["software"].items():
        lines.append(
            f"| {_markdown_cell(component)} | {_markdown_cell(version)} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in _LIMITATIONS)
    return "\n".join(lines) + "\n"


def _append_markdown_table(
    lines: list[str],
    headers: Sequence[object],
    rows: Sequence[Sequence[object]],
) -> None:
    if not headers:
        raise ReportError("Markdown table must have at least one column")
    width = len(headers)
    lines.append("| " + " | ".join(_markdown_cell(value) for value in headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        if len(row) != width:
            raise ReportError("Markdown table row has the wrong column count")
        lines.append(
            "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        )


def _flatten_markdown_fields(
    value: object, prefix: str = ""
) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        if not value:
            return [(prefix or "value", "none")]
        flattened: list[tuple[str, object]] = []
        for key, child in value.items():
            child_prefix = f"{prefix} / {key}" if prefix else str(key)
            flattened.extend(_flatten_markdown_fields(child, child_prefix))
        return flattened
    if isinstance(value, (list, tuple)):
        if not value:
            return [(prefix or "value", "none")]
        flattened = []
        for index, child in enumerate(value, 1):
            child_prefix = (
                f"{prefix} / item {index}" if prefix else f"item {index}"
            )
            flattened.extend(_flatten_markdown_fields(child, child_prefix))
        return flattened
    return [(prefix or "value", value)]


def _markdown_cell(value: object) -> str:
    if isinstance(value, Mapping) or isinstance(
        value, (list, tuple, set, frozenset, np.ndarray)
    ):
        raise ReportError("Markdown table cells must contain scalar values")
    if value is None:
        text = "none"
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ReportError("Markdown table cells must contain finite values")
        text = f"{value:.12g}"
    else:
        text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return (
        text.replace("&", "&amp;")
        .replace("|", "&#124;")
        .replace("`", "&#96;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("\n", "<br>")
    )


def _render_pdf(
    context: ReportContext,
    data: _ReportData,
    path: Path | _OwnedWriteHandle,
) -> None:
    payload = data.payload
    footer = (
        "Research use only — not a clinical diagnosis — acquisition parameters "
        "not independently verified — visual review NOT_REVIEWED"
    )
    metadata = {
        "Title": "Diffusion MRI technical QC report",
        "Author": "dmri-reproducible-pipeline",
        "Subject": "Research technical QC",
        "Keywords": "dMRI QC",
    }
    with PdfPages(path, metadata=metadata) as pdf:
        figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        axis = figure.add_subplot(111)
        axis.axis("off")
        values = payload["global_values"]
        global_text = "\n".join(
            f"{key}: {float(value):.8g}" for key, value in values.items()
        )
        shell_text = "\n".join(
            f"  {key}: {value}"
            for key, value in payload["input"]["shell_distribution"].items()
        )
        axis.text(
            0.02,
            0.97,
            f"Technical QC headline — {payload['subject_id']}\n"
            "Processing status: COMPLETED\nvisual_review_status: NOT_REVIEWED\n\n"
            "PA/AP phase-encoding vectors: "
            f"{_format_sequence(payload['acquisition_assumptions']['pa_phase_encoding_vector'])} / "
            f"{_format_sequence(payload['acquisition_assumptions']['ap_phase_encoding_vector'])}; "
            "TotalReadoutTime "
            f"{payload['acquisition_assumptions']['total_readout_time_seconds']} s\n"
            "User-supplied; not independently verified from DICOM or scanner JSON.\n"
            f"PA/AP shapes: {_format_sequence(payload['input']['pa_shape'])} / "
            f"{_format_sequence(payload['input']['ap_shape'])}; voxel sizes (mm): "
            f"{_format_sequence(payload['input']['pa_voxel_size_mm'])} / "
            f"{_format_sequence(payload['input']['ap_voxel_size_mm'])}\n"
            f"Shell counts:\n{shell_text}\n\n"
            f"All 17 global values\n{global_text}",
            va="top",
            family="monospace",
            fontsize=9,
        )
        _pdf_footer(figure, footer)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(
            1, 2, figsize=(11.69, 8.27), constrained_layout=True
        )
        axes[0].imshow(_report_figure(data, "stripe"))
        axes[0].axis("off")
        axes[0].set_title("Pre-denoise stripe screen")
        axes[1].axis("off")
        high = _format_sequence(
            payload["pre_denoise_motion_qc"]["high_volumes_one_based"]
        )
        ambiguous = _format_sequence(
            payload["pre_denoise_motion_qc"]["ambiguous_volumes_one_based"]
        )
        axes[1].text(
            0,
            0.95,
            f"Decision: {payload['pre_denoise_motion_qc']['decision']}\n"
            f"High count: {payload['pre_denoise_motion_qc']['high_volume_count']}\n"
            f"Ambiguous count: {payload['pre_denoise_motion_qc']['ambiguous_volume_count']}\n"
            f"Maximum cSI: {payload['pre_denoise_motion_qc']['maximum_csi']:.6g}\n"
            f"One-based high: {high}\n"
            f"One-based ambiguous: {ambiguous}\n\n"
            "Detail sheets:\n"
            + "\n".join(item.name for item in data.detail_files),
            va="top",
        )
        _pdf_footer(figure, footer)
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        axis = figure.add_subplot(111)
        axis.imshow(_report_figure(data, "overview"))
        axis.axis("off")
        axis.set_title("Complete stepwise overview")
        _pdf_footer(figure, footer)
        pdf.savefig(figure)
        plt.close(figure)

        figure = _make_pdf_page4(data, footer)
        pdf.savefig(figure)
        plt.close(figure)

        figure = _make_pdf_page5(data, footer)
        pdf.savefig(figure)
        plt.close(figure)

        figure = _make_pdf_page6(data, footer)
        pdf.savefig(figure)
        plt.close(figure)


def _report_figure(data: _ReportData, figure_id: str) -> np.ndarray:
    try:
        original = data.figures[figure_id]
        snapshot = data.snapshots[original]
    except KeyError as error:
        raise ReportError(f"missing immutable report figure {figure_id}") from error
    return mpimg.imread(snapshot)


def _format_sequence(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)


def _format_warnings(value: object) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "Warnings: none"
    return "Warnings:\n" + "\n".join(f"• {item}" for item in value)


def _format_error_histogram(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        raise ReportError("NODDI error histogram is not a sequence")
    lines = ["Cleaned-mask error codes:"]
    visible = value[:2]
    summaries = []
    for entry in visible:
        if not isinstance(entry, Mapping):
            raise ReportError("NODDI error histogram entry is invalid")
        summaries.append(
            f"code {entry['error_code']}: {entry['voxel_count']} voxels"
        )
    if summaries:
        lines.append("  " + "; ".join(summaries))
    if len(value) > len(visible):
        lines.append(
            f"  … {len(value) - len(visible)} additional codes; see JSON"
        )
    return "\n".join(lines)


def _format_quad(value: object) -> str:
    if not isinstance(value, Mapping):
        raise ReportError("EDDY QUAD summary is not a mapping")
    if not value:
        return "Selected EDDY QUAD metrics: none"
    items = sorted(value.items())
    visible = items[:6]
    lines = [
        f"  {key}: {float(number):.6g}" for key, number in visible
    ]
    if len(items) > len(visible):
        lines.append(f"  … {len(items) - len(visible)} additional metrics; see JSON")
    return "Selected EDDY QUAD metrics:\n" + "\n".join(lines)


def _pdf_page4_summary(payload: Mapping[str, object]) -> tuple[str, str, str]:
    topup = payload["topup"]
    bet = payload["bet"]
    eddy = payload["eddy"]
    runtime = eddy["runtime_seconds"]
    if not isinstance(runtime, Mapping):
        raise ReportError("EDDY runtime summary is not a mapping")
    return (
        "TOPUP FIELD QC\n"
        "Units: Hz\n"
        f"Brain-mask voxels: {topup['brain_mask_voxel_count']}\n"
        f"Median: {topup['field_median']:.6g}\n"
        f"1st percentile: {topup['field_p01']:.6g}\n"
        f"99th percentile: {topup['field_p99']:.6g}",
        "BET MASK QC\n"
        f"Components: {bet['component_count']}\n"
        f"Original voxels: {bet['original_voxel_count']}\n"
        f"Largest component: {bet['largest_voxel_count']}\n"
        f"Removed voxels: {bet['removed_voxel_count']}\n"
        f"Largest-component tie: {bet['largest_tie']}\n"
        f"Tie warning: {bet['tie_warning'] or 'none'}",
        "EDDY MOTION / OUTLIER QC\n"
        f"Max translation: {eddy['translation_max_abs_mm']:.6g} mm\n"
        f"Max rotation: {eddy['rotation_max_abs_degrees']:.6g} degrees\n"
        f"Absolute RMS mean/max: {eddy['absolute_rms_mean_mm']:.6g} / "
        f"{eddy['absolute_rms_max_mm']:.6g} mm\n"
        f"Relative RMS mean/max: {eddy['relative_rms_mean_mm']:.6g} / "
        f"{eddy['relative_rms_max_mm']:.6g} mm\n"
        "Outlier slices reported/observed: "
        f"{eddy['reported_outlier_slice_count']} / "
        f"{eddy['observed_outlier_slice_count']}\n"
        f"Affected volumes: {eddy['affected_volume_count']}\n"
        "Max slices in one volume: "
        f"{eddy['maximum_slices_in_one_volume']}\n"
        "EDDY command (includes CNR maps and residuals): "
        f"{runtime['eddy_command_including_cnr_and_residuals']:.6g} seconds\n"
        f"EDDY QUAD: {runtime['eddy_quad']:.6g} seconds\n"
        "05_eddy stage action total: "
        f"{runtime['stage_action_total']:.6g} seconds\n"
        f"{_format_quad(eddy['eddy_quad_selected_numeric_metrics'])}",
    )


def _model_summary(title: str, facts: Mapping[str, object]) -> str:
    lines = [title]
    for key, label in (
        ("model", "Model"),
        ("model_name", "Model"),
        ("selection", "Selection"),
        ("selected_volume_count", "Selected volumes"),
        ("selected_b0_count", "Selected b0 volumes"),
        ("shells", "Shells"),
        ("worker_count", "Workers"),
        ("success_count", "Successful voxels"),
        ("error_999_count", "Error 999 voxels"),
        ("other_error_count", "Other error voxels"),
    ):
        if key in facts:
            lines.append(f"{label}: {_format_sequence(facts[key])}")
    lines.append(_format_warnings(facts.get("warnings")))
    return "\n".join(lines)


def _pdf_page5_summary(payload: Mapping[str, object]) -> tuple[str, str, str, str]:
    models = payload["models"]
    noddi = _model_summary("NODDI", models["noddi"])
    noddi += "\n" + _format_error_histogram(
        payload["noddi"]["error_code_histogram"]
    )
    return (
        _model_summary("DTI", models["dti"]),
        _model_summary("DKI", models["dki"]),
        _model_summary("DIRECT DKI", models["dki_direct"]),
        noddi,
    )


def _make_pdf_page4(data: _ReportData, footer: str) -> plt.Figure:
    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.985,
        bottom=0.075,
        top=0.95,
        hspace=0.04,
        wspace=0.035,
        height_ratios=(0.36, 0.64),
    )
    summaries = _pdf_page4_summary(data.payload)
    figure_ids = ("topup", "bet", "eddy_motion")
    artists: list[tuple[plt.Text, plt.Axes]] = []
    for column, (summary, figure_id) in enumerate(
        zip(summaries, figure_ids, strict=True)
    ):
        text_axis = figure.add_subplot(grid[0, column])
        text_axis.axis("off")
        artists.append(
            (
                text_axis.text(
                0.02,
                0.98,
                summary,
                transform=text_axis.transAxes,
                va="top",
                ha="left",
                fontsize=6.8 if figure_id == "eddy_motion" else 7.5,
                linespacing=1.2,
                ),
                text_axis,
            )
        )
        image_axis = figure.add_subplot(grid[1, column])
        image_axis.imshow(_report_figure(data, figure_id))
        image_axis.axis("off")
    _pdf_footer(figure, footer)
    _assert_artists_within_axes(figure, artists)
    return figure


def _make_pdf_page5(data: _ReportData, footer: str) -> plt.Figure:
    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.985,
        bottom=0.07,
        top=0.965,
        hspace=0.08,
        wspace=0.055,
    )
    summaries = _pdf_page5_summary(data.payload)
    figure_ids = ("dti", "dki", "dki_direct", "noddi")
    artists: list[tuple[plt.Text, plt.Axes]] = []
    for index, (summary, figure_id) in enumerate(
        zip(summaries, figure_ids, strict=True)
    ):
        nested = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=(0.31, 0.69), hspace=0.02
        )
        text_axis = figure.add_subplot(nested[0])
        text_axis.axis("off")
        artists.append(
            (
                text_axis.text(
                0.01,
                0.98,
                summary,
                transform=text_axis.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                linespacing=1.15,
                ),
                text_axis,
            )
        )
        image_axis = figure.add_subplot(nested[1])
        image_axis.imshow(_report_figure(data, figure_id))
        image_axis.axis("off")
    _pdf_footer(figure, footer)
    _assert_artists_within_axes(figure, artists)
    return figure


def _stage_output_hash_aggregate(stage: Mapping[str, object]) -> str:
    ordered = [entry["sha256"] for entry in stage["outputs"]]
    encoded = json.dumps(
        ordered, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pdf_page6_provenance(
    payload: Mapping[str, object],
) -> tuple[str, ...]:
    blocks = []
    for stage in payload["stages"]:
        blocks.append(
            f"{stage['stage']}\n"
            f"  started: {stage['started_utc']}\n"
            f"  completed: {stage['completed_utc']}\n"
            f"  stage signature: {stage['stage_signature']}\n"
            f"  output hash aggregate: {_stage_output_hash_aggregate(stage)}"
        )
    if tuple(block.splitlines()[0] for block in blocks) != REPORT_STAGE_ORDER:
        raise ReportError("PDF stage provenance order is invalid")
    return tuple(blocks)


def _make_pdf_page6(data: _ReportData, footer: str) -> plt.Figure:
    payload = data.payload
    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=False)
    outer = figure.add_gridspec(
        1,
        2,
        left=0.035,
        right=0.985,
        bottom=0.07,
        top=0.96,
        wspace=0.045,
        width_ratios=(0.37, 0.63),
    )
    left = outer[0].subgridspec(2, 1, height_ratios=(0.49, 0.51), hspace=0.035)
    image_axis = figure.add_subplot(left[0])
    image_axis.imshow(_report_figure(data, "jhu_48roi"))
    image_axis.axis("off")
    image_axis.set_title("JHU exact 48-label alignment", fontsize=9)

    resource_lines = []
    for name, details in payload["atlas"]["resource_sha256"].items():
        if not isinstance(details, Mapping) or not isinstance(
            details.get("sha256"), str
        ):
            raise ReportError("atlas resource SHA-256 evidence is invalid")
        digest = details["sha256"]
        resource_lines.extend(
            (f"  {name}:", f"    {digest[:32]}", f"    {digest[32:]}")
        )
    resources = "\n".join(resource_lines)
    mask_counts = "\n".join(
        f"  {name}: {count}"
        for name, count in payload["summary_mask"].items()
    )
    left_text = (
        f"Atlas: {payload['atlas']['name']}\n"
        f"Version: {payload['atlas']['version']}\n"
        f"Resource SHA-256:\n{resources}\n"
        "Warped atlas SHA-256:\n"
        f"  {payload['atlas']['warped_atlas_sha256'][:32]}\n"
        f"  {payload['atlas']['warped_atlas_sha256'][32:]}\n"
        "Nonzero labels: 1–48\n"
        f"ROI rows: {payload['roi_summary']['row_count']}\n"
        f"ROI voxel total: {payload['roi_summary']['voxel_count_total']}\n"
        f"Summary-mask counts:\n{mask_counts}\n\n"
        "Limitations:\n"
        + "\n".join(f"• {item}" for item in _LIMITATIONS)
    )
    facts_axis = figure.add_subplot(left[1])
    facts_axis.axis("off")
    left_artist = facts_axis.text(
        0.01,
        0.99,
        left_text,
        transform=facts_axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=5.3,
        linespacing=1.1,
    )

    provenance_axis = figure.add_subplot(outer[1])
    provenance_axis.axis("off")
    provenance_text = (
        "ORDERED STAGE PROVENANCE — full SHA-256 evidence\n\n"
        + "\n\n".join(_pdf_page6_provenance(payload))
    )
    provenance_artist = provenance_axis.text(
        0.0,
        1.0,
        provenance_text,
        transform=provenance_axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=4.45,
        linespacing=1.02,
    )
    _pdf_footer(figure, footer)
    _assert_artists_within_figure(
        figure, (left_artist, provenance_artist)
    )
    return figure


def _assert_artists_within_figure(
    figure: plt.Figure, artists: Sequence[plt.Artist]
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = figure.bbox
    tolerance = 1.0
    for artist in artists:
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < frame.x0 - tolerance
            or bounds.y0 < frame.y0 - tolerance
            or bounds.x1 > frame.x1 + tolerance
            or bounds.y1 > frame.y1 + tolerance
        ):
            raise ReportError("PDF summary text extends beyond the page boundary")


def _assert_artists_within_axes(
    figure: plt.Figure,
    artists: Sequence[tuple[plt.Artist, plt.Axes]],
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tolerance = 1.0
    for artist, axis in artists:
        bounds = artist.get_window_extent(renderer=renderer)
        panel = axis.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < panel.x0 - tolerance
            or bounds.y0 < panel.y0 - tolerance
            or bounds.x1 > panel.x1 + tolerance
            or bounds.y1 > panel.y1 + tolerance
        ):
            raise ReportError("PDF summary text extends beyond its assigned panel")


def _pdf_footer(figure: plt.Figure, text: str) -> None:
    figure.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=7)


def _validate_pdf_bytes(data: bytes) -> None:
    if (
        not data.startswith(b"%PDF-")
        or not data.rstrip().endswith(b"%%EOF")
        or len(data) < 1000
    ):
        raise ReportError("generated PDF framing is invalid")
    pages = len(re.findall(rb"/Type\s*/Page(?!s)\b", data))
    if pages != 6:
        raise ReportError(f"generated PDF must contain exactly six pages, found {pages}")


def _validate_markdown_links(
    text: str,
    context: ReportContext,
    figures: Mapping[str, Path],
) -> None:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    expected_targets = [
        *figures.values(),
        *context.stripe_detail_files,
        context.global_csv,
        context.roi_csv,
        context.summary_json,
    ]
    allowed = {_safe_link(context, target): target for target in expected_targets}
    for value in links:
        if "://" in value or Path(value).is_absolute():
            raise ReportError("Markdown links must be safe subject-relative paths")
        if value not in allowed:
            raise ReportError(f"Markdown link is not an assigned report input: {value}")
        if not allowed[value].exists():
            raise ReportError(f"Markdown link does not resolve: {value}")


def _sanitize_metrics(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return _sanitize_string(value) if isinstance(value, str) else value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ReportError("metrics JSON contains non-finite values")
        return value
    if isinstance(value, list):
        return [_sanitize_metrics(child) for child in value]
    if isinstance(value, dict):
        result: OrderedDict[str, object] = OrderedDict()
        for key in sorted(value):
            if any(token in key.lower() for token in ("path", "file", "dir", "command")):
                continue
            result[key] = _sanitize_metrics(value[key])
        return result
    raise ReportError("metrics JSON contains an unsupported value type")


def _sanitize_software(value: str) -> str:
    return _sanitize_string(value)


def _sanitize_string(value: str) -> str:
    if _contains_machine_path(value) or ".work" in value:
        return "[path removed]"
    return value


def _contains_machine_path(value: str) -> bool:
    uri = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://")
    windows = re.compile(
        r"""(?ix)(?<![\w.])(?:[a-z]:[\\/]|\\\\)[^\s"'<>|]+"""
    )
    unc = re.compile(r"""(?x)(?<![\w./])//[^\s/]+/[^\s"'<>|]+""")
    escaped_posix = re.compile(
        r"""(?ix)(?<![\w./])/(?:\\u[0-9a-f]{4})+"""
    )
    posix = re.compile(
        r"""(?x)(?<![\w./])/(?![/\s])[^/\s\\'"<>|)\]}]+"""
        r"""(?:/[^/\s\\'"<>|)\]}]+)*"""
    )
    return bool(
        uri.search(value)
        or windows.search(value)
        or unc.search(value)
        or escaped_posix.search(value)
        or posix.search(value)
    )


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        return _load_json(path, label)
    except QCError as error:
        raise ReportError(str(error)) from error


def _read_csv(path: Path) -> list[OrderedDict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ReportError("CSV header is missing or duplicated")
            return [
                OrderedDict((field, row[field]) for field in reader.fieldnames)
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReportError("cannot read report CSV input") from error


def _load_table(path: Path, label: str) -> np.ndarray:
    try:
        return _load_numeric_table(path, label)
    except QCError as error:
        raise ReportError(str(error)) from error


def _load_3d(
    path: Path, label: str
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    try:
        image = nib.load(path, mmap=True)
        values = np.asarray(image.dataobj, dtype=np.float64)
    except (OSError, ValueError, TypeError) as error:
        raise ReportError(f"cannot read {label} NIfTI") from error
    if (
        len(image.shape) != 3
        or not np.isfinite(image.affine).all()
        or not np.isfinite(values).all()
    ):
        raise ReportError(f"{label} must be a finite 3D NIfTI")
    return image, values


def _same_grid(
    reference: nib.spatialimages.SpatialImage,
    image: nib.spatialimages.SpatialImage,
    label: str,
) -> None:
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-5, rtol=0.0
    ):
        raise ReportError(f"{label} must share the cleaned brain-mask grid")


def _require_empty_output(path: Path) -> _DirectoryAnchor:
    _reject_path(path)
    try:
        return _pin_output_directory(
            path,
            require_empty=True,
            label="report output directory",
        )
    except QCError as error:
        raise ReportError(str(error)) from error


def _output_paths(context: ReportContext) -> ReportOutputs:
    subject = context.stage_context.config.subject_id
    return ReportOutputs(
        pdf=context.output_directory / f"{subject}_QC_report.pdf",
        markdown=context.output_directory / f"{subject}_analysis_report.md",
        run_summary_json=context.output_directory / f"{subject}_run_summary.json",
    )


def _safe_link(context: ReportContext, path: Path) -> str:
    _relative_to_subject(path, context.stage_context.subject_root)
    logical_output = _logical_promoted_path(
        context.output_directory, context.stage_context.subject_root
    )
    logical_target = _logical_promoted_path(
        path, context.stage_context.subject_root
    )
    value = os.path.relpath(logical_target, logical_output)
    candidate = (logical_output / value).resolve(strict=False)
    try:
        candidate.relative_to(context.stage_context.subject_root.resolve(strict=False))
    except ValueError as error:
        raise ReportError("report relative link escapes subject output") from error
    if Path(value).is_absolute() or "://" in value or ".work" in Path(value).parts:
        raise ReportError("report relative link is unsafe")
    return Path(value).as_posix()


def _logical_promoted_path(path: Path, subject_root: Path) -> Path:
    relative = _relative_to_subject(path, subject_root, require_exists=False)
    if len(relative.parts) >= 2 and relative.parts[0] == ".work":
        relative = Path(*relative.parts[1:])
    return subject_root.resolve(strict=False) / relative


def _safe_record_relative(value: str) -> None:
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or ".work" in path.parts
        or "://" in value
        or not path.parts
    ):
        raise ReportError("stage record contains an unsafe output path")


def _validate_serialized_report(text: str) -> None:
    if _contains_machine_path(text) or ".work" in text:
        raise ReportError("report contains a private path, URI, or work path")


def _finite_recursive(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportError(f"{label} contains NaN or Infinity")
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_recursive(child, label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_recursive(child, label)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return OrderedDict((str(key), _json_ready(child)) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _validate_png_report(path: Path) -> tuple[int, int]:
    try:
        return _validate_png(path)
    except QCError as error:
        raise ReportError(str(error)) from error


def _require_regular(path: Path) -> None:
    _reject_path(path)
    try:
        details = path.lstat()
    except OSError as error:
        raise ReportError(f"report input is not readable: {path.name}") from error
    if not stat.S_ISREG(details.st_mode):
        raise ReportError(f"report input must be a regular file: {path.name}")
    if details.st_nlink != 1:
        raise ReportError(f"report input must have link count 1: {path.name}")


def _reject_path(path: Path) -> None:
    if ".." in path.parts:
        raise ReportError("report paths must not contain parent traversal")
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ReportError("report paths must not contain symbolic-link components")


def _relative_to_subject(
    path: Path, subject_root: Path, *, require_exists: bool = True
) -> Path:
    try:
        relative = path.resolve(strict=require_exists).relative_to(
            subject_root.resolve(strict=False)
        )
    except (OSError, ValueError) as error:
        raise ReportError("report subject paths must stay within subject output") from error
    if not relative.parts:
        raise ReportError("report path must name an item within subject output")
    return relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ReportError(f"cannot hash report input: {path.name}") from error
    return digest.hexdigest()


__all__ = [
    "REPORT_JSON_KEYS",
    "REPORT_STAGE_ORDER",
    "ReportContext",
    "ReportError",
    "ReportOutputs",
    "STAGE_METRIC_KEYS",
    "write_final_report",
]
