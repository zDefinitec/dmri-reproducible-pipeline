from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dmri_pipeline.audit import audit_inputs, write_input_audit
from dmri_pipeline.report import (
    REPORT_JSON_KEYS,
    ReportContext,
    ReportError,
    write_final_report,
)
from dmri_pipeline.state import StageRecord
from dmri_pipeline.summary import SummaryContext, summarize_subject
from test_qc import SyntheticQCCase, make_qc_case


@dataclass(frozen=True)
class SyntheticReportCase:
    context: ReportContext
    qc: SyntheticQCCase


def make_report_case(subject_config) -> SyntheticReportCase:
    qc = make_qc_case(subject_config)
    from dmri_pipeline.qc import generate_all_qc

    qc_figures = generate_all_qc(qc.context)
    subject_root = subject_config.subject_output
    input_audit = subject_root / "input_audit.json"
    write_input_audit(audit_inputs(subject_config), input_audit)
    detail = (
        qc.context.stripe_metrics_csv.parent
        / "03_candidate_details_01_01.png"
    )
    shutil.copyfile(qc_figures["stripe"], detail)

    metrics_dir = subject_root / "metrics"
    metrics_dir.mkdir()

    def metric(name: str, payload: dict[str, object]) -> Path:
        path = metrics_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    stage_metrics = {
        "denoise": metric(
            "denoise",
            {
                "pa_measurement_count": 8,
                "ap_measurement_count": 2,
                "pa_patch_radius": 1,
            },
        ),
        "gibbs": metric("gibbs", {"slice_axis": 2, "n_points": 3}),
        "topup": metric("topup", {"pa_b0_count": 1, "ap_b0_count": 2}),
        "bet": metric(
            "bet",
            {
                "component_count": 1,
                "original_voxel_count": 220,
                "largest_voxel_count": 216,
                "removed_voxel_count": 4,
                "largest_tie": False,
                "selected_component": 1,
            },
        ),
        "dti": metric(
            "dti",
            {
                "model": "DIPY TensorModel NLLS",
                "selection": "b <= 1200 s/mm^2",
                "selected_volume_count": 5,
                "selected_b0_count": 1,
                "nonfinite_replaced": {
                    "FA": {"inside_brain": 1, "outside_brain": 0}
                },
            },
        ),
        "dki": metric(
            "dki",
            {
                "model": "DIPY DiffusionKurtosisModel WLS",
                "selection": "all acquired shells",
                "selected_volume_count": 8,
                "shells": [0, 200, 500, 1000, 2000, 3000],
                "nonfinite_replaced": {},
            },
        ),
        "dki_direct": metric(
            "dki_direct",
            {
                "model": "Henrique avs_dki_df average-signal direct fit",
                "selection": "all acquired shells",
                "selected_volume_count": 8,
                "nonfinite_replaced": {
                    "MK": {"inside_brain": 2, "outside_brain": 0}
                },
            },
        ),
        "noddi": metric(
            "noddi",
            {
                "model_name": "NODDI WatsonSHStickTortIsoV_B0",
                "worker_count": 2,
                "success_count": 215,
                "error_999_count": 0,
                "other_error_count": 1,
            },
        ),
    }
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    topup_field = subject_root / "topup_field_hz.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.linspace(-10, 10, 512, dtype=np.float32).reshape(8, 8, 8),
            affine,
        ),
        topup_field,
    )
    errors = np.zeros((8, 8, 8), dtype=np.int16)
    brain_values = np.asarray(
        nib.load(qc.context.brain_mask).dataobj, dtype=np.uint8
    )
    nonatlas_brain_coordinate = np.argwhere(brain_values == 1)[100]
    errors[tuple(nonatlas_brain_coordinate)] = 1
    noddi_errors = subject_root / "noddi_error_codes.nii.gz"
    nib.save(nib.Nifti1Image(errors, affine), noddi_errors)
    outlier_report = subject_root / "eddy_outlier_report.txt"
    outlier_report.write_text(
        "Slice 0 in scan 1 is an outlier with mean 4.2 standard deviations off\n"
        "Slice 1 in scan 2 is an outlier with mean 4.6 standard deviations off\n"
        "Slice 2 in scan 2 is an outlier with mean 5.1 standard deviations off\n",
        encoding="utf-8",
    )
    eddy_quad = subject_root / "eddy_quad.json"
    eddy_quad.write_text(
        json.dumps(
            {
                "qc_motion_absolute": 0.22,
                "qc_motion_relative": 0.11,
                "source_path": "/" + "private/clinical/input",
                "nested": {"cnr_mean": 4.5, "name": "ignored"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_dir = subject_root / "10_summary"
    package_root = Path(__file__).parents[1]
    summary = summarize_subject(
        SummaryContext(
            config=subject_config,
            warped_atlas=qc.context.warped_atlas,
            brain_mask=qc.context.brain_mask,
            metric_maps={
                "DTI_FA": qc.context.dti_maps["FA"],
                "DTI_MD": qc.context.dti_maps["MD"],
                "DTI_AD": qc.context.dti_maps["AD"],
                "DTI_RD": qc.context.dti_maps["RD"],
                "DKI_FA": qc.context.dki_maps["FA"],
                "DKI_MD": qc.context.dki_maps["MD"],
                "DKI_AD": qc.context.dki_maps["AD"],
                "DKI_RD": qc.context.dki_maps["RD"],
                "DKI_MK": qc.context.dki_maps["MK"],
                "DKI_AK": qc.context.dki_maps["AK"],
                "DKI_RK": qc.context.dki_maps["RK"],
                "DKI_DIRECT_MD": qc.context.dki_direct_maps["MD"],
                "DKI_DIRECT_MK": qc.context.dki_direct_maps["MK"],
                "DKI_DIRECT_S0": qc.context.dki_direct_maps["S0"],
                "NODDI_ODI": qc.context.noddi_maps["ODI"],
                "NODDI_FICVF": qc.context.noddi_maps["FICVF"],
                "NODDI_FISO": qc.context.noddi_maps["FISO"],
            },
            noddi_error_codes=noddi_errors,
            atlas_xml=package_root / "resources/jhu_48roi/JHU-labels.xml",
            atlas_provenance=package_root
            / "resources/jhu_48roi/provenance.json",
            output_directory=summary_dir,
        )
    )

    records_dir = subject_root / "records"
    records_dir.mkdir()
    stage_names = (
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
    stage_records: list[Path] = []
    for index, stage in enumerate(stage_names):
        digest = f"{index + 1:064x}"[-64:]
        record = StageRecord(
            stage=stage,
            subject_id=subject_config.subject_id,
            package_version="1.0.0",
            config_sha256="a" * 64,
            stage_signature=digest,
            started_utc=f"2026-01-{index + 1:02d}T00:00:00.000000Z",
            completed_utc=f"2026-01-{index + 1:02d}T00:01:00.000000Z",
            inputs=({"path": "redacted-input", "sha256": "b" * 64},),
            outputs=(
                {
                    "relative_path": f"{stage}/artifact.txt",
                    "sha256": "c" * 64,
                    "size": index + 1,
                },
            ),
            software={"python": "3.11", "stage-tool": "synthetic-1"},
        )
        path = records_dir / f"{index:02d}_{stage}.json"
        path.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage_records.append(path)

    report_output = subject_root / "report"
    report_output.mkdir()
    context = ReportContext(
        stage_context=qc.context.stage_context,
        output_directory=report_output,
        qc_manifest_json=qc.context.output_directory / "qc_manifest.json",
        input_audit_json=input_audit,
        stripe_metrics_csv=qc.context.stripe_metrics_csv,
        stripe_decision_json=qc.context.stripe_decision_json,
        stripe_detail_files=(detail,),
        stage_metrics_json=stage_metrics,
        topup_field_hz=topup_field,
        brain_mask=qc.context.brain_mask,
        eddy_parameters=qc.context.eddy_parameters,
        eddy_movement_rms=qc.context.eddy_movement_rms,
        eddy_outlier_map=qc.context.eddy_outlier_map,
        eddy_outlier_report=outlier_report,
        eddy_quad_json=eddy_quad,
        noddi_error_codes=noddi_errors,
        summary_json=summary.summary_json,
        global_csv=summary.global_csv,
        roi_csv=summary.roi_csv,
        atlas_provenance_json=package_root
        / "resources/jhu_48roi/provenance.json",
        stage_records=tuple(stage_records),
    )
    return SyntheticReportCase(context, qc)


@pytest.fixture
def report_case(subject_config) -> SyntheticReportCase:
    return make_report_case(subject_config)


def test_report_schema_contract_is_frozen() -> None:
    assert REPORT_JSON_KEYS == (
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


def test_final_report_has_exact_outputs_schema_six_pages_and_safe_links(
    report_case: SyntheticReportCase,
) -> None:
    outputs = write_final_report(report_case.context)
    assert outputs.pdf.name == "SYNTH001_QC_report.pdf"
    assert outputs.markdown.name == "SYNTH001_analysis_report.md"
    assert outputs.run_summary_json.name == "SYNTH001_run_summary.json"
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.__dict__.values())

    payload = json.loads(outputs.run_summary_json.read_text(encoding="utf-8"))
    assert tuple(payload) == REPORT_JSON_KEYS
    assert payload["visual_review_status"] == "NOT_REVIEWED"
    assert payload["processing_status"] == "COMPLETED"
    assert len(payload["global_values"]) == 17
    assert payload["roi_summary"]["row_count"] == 48
    assert payload["roi_summary"]["true_label_ids"] == list(range(1, 49))
    assert payload["atlas"]["nonzero_labels"] == list(range(1, 49))
    assert payload["eddy"]["observed_outlier_slice_count"] == 3
    assert payload["eddy"]["reported_outlier_slice_count"] == 3
    assert payload["eddy"]["affected_volume_count"] == 2
    assert payload["noddi"]["error_code_histogram"] == [
        {"error_code": 0, "voxel_count": 215},
        {"error_code": 1, "voxel_count": 1},
    ]
    assert payload["models"]["dti"]["warnings"]
    assert payload["models"]["dki_direct"]["warnings"]
    assert "source_path" not in json.dumps(payload["eddy"])
    assert len(payload["stages"]) == 14

    serialized = json.dumps(payload)
    assert "/Users/" not in serialized
    assert ("/" + "private/clinical") not in serialized
    assert ".work" not in serialized
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    for link in payload["outputs"]["qc_figures"].values():
        assert (report_case.context.output_directory / link).resolve().exists()

    markdown = outputs.markdown.read_text(encoding="utf-8")
    assert "Research use only" in markdown
    assert "not a clinical diagnosis" in markdown
    assert "NOT_REVIEWED" in markdown
    assert "not inferred or independently verified" in markdown
    assert "- High volume count: `1`" in markdown
    assert "- Ambiguous volume count: `1`" in markdown
    assert "- Maximum outlier slices in one volume: `2`" in markdown
    assert "17 global" not in markdown  # values are listed explicitly, not summarized away
    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown):
        assert (report_case.context.output_directory / link).resolve().exists()

    pdf = outputs.pdf.read_bytes()
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(re.findall(rb"/Type\s*/Page(?!s)\b", pdf)) == 6
    assert b"OrderedDict" not in pdf
    assert b"mappingproxy" not in pdf


def test_markdown_and_json_are_deterministic_for_identical_inputs(
    report_case: SyntheticReportCase,
) -> None:
    first = write_final_report(report_case.context)
    second_context = replace(
        report_case.context,
        output_directory=report_case.context.stage_context.subject_root / "report_again",
    )
    second_context.output_directory.mkdir()
    second = write_final_report(second_context)
    assert first.markdown.read_bytes() == second.markdown.read_bytes()
    assert first.run_summary_json.read_bytes() == second.run_summary_json.read_bytes()


def test_report_workdir_links_target_promoted_subject_layout(
    report_case: SyntheticReportCase,
) -> None:
    work_context = replace(
        report_case.context,
        output_directory=report_case.context.stage_context.subject_root
        / ".work"
        / "report",
    )
    work_context.output_directory.mkdir(parents=True)
    outputs = write_final_report(work_context)
    payload = json.loads(outputs.run_summary_json.read_text(encoding="utf-8"))
    assert payload["outputs"]["qc_figures"]["input"] == "../qc/00_input_b0.png"
    serialized = outputs.run_summary_json.read_text(encoding="utf-8")
    assert ".work" not in serialized
    assert "../../qc" not in serialized


def test_report_rejects_wrong_roi_count_and_nonempty_output(
    report_case: SyntheticReportCase,
) -> None:
    lines = report_case.context.roi_csv.read_text(encoding="utf-8").splitlines()
    report_case.context.roi_csv.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ReportError, match="48"):
        write_final_report(report_case.context)
    report_case.context.output_directory.mkdir(exist_ok=True)
    (report_case.context.output_directory / "foreign.txt").write_text(
        "foreign\n", encoding="utf-8"
    )
    with pytest.raises(ReportError, match="empty"):
        write_final_report(report_case.context)


def test_report_detects_input_mutation_and_leaves_no_partial_outputs(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._render_pdf

    def mutate(context, data, path):
        original(context, data, path)
        context.global_csv.write_text(
            context.global_csv.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(report_module, "_render_pdf", mutate)
    with pytest.raises(ReportError, match="changed"):
        write_final_report(report_case.context)
    assert list(report_case.context.output_directory.iterdir()) == []


def test_report_context_rejects_extra_stage_metric_and_symlink(
    report_case: SyntheticReportCase, tmp_path: Path
) -> None:
    with pytest.raises(ReportError, match="exact keys"):
        ReportContext(
            **{
                **report_case.context.__dict__,
                "stage_metrics_json": {
                    **report_case.context.stage_metrics_json,
                    "extra": report_case.context.input_audit_json,
                },
            }
        )
    link = tmp_path / "summary_link.json"
    link.symlink_to(report_case.context.summary_json)
    with pytest.raises(ReportError, match="symbolic"):
        ReportContext(**{**report_case.context.__dict__, "summary_json": link})


def test_report_rejects_missing_or_reordered_stage_records(
    report_case: SyntheticReportCase,
) -> None:
    missing = replace(
        report_case.context,
        stage_records=report_case.context.stage_records[:-1],
    )
    with pytest.raises(ReportError, match="every completed stage"):
        write_final_report(missing)
    reordered = replace(
        report_case.context,
        output_directory=report_case.context.stage_context.subject_root
        / "report_reordered",
        stage_records=(
            *report_case.context.stage_records[:-2],
            report_case.context.stage_records[-1],
            report_case.context.stage_records[-2],
        ),
    )
    reordered.output_directory.mkdir()
    with pytest.raises(ReportError, match="out of order"):
        write_final_report(reordered)


def test_report_rejects_duplicate_json_keys(
    report_case: SyntheticReportCase,
) -> None:
    denoise = report_case.context.stage_metrics_json["denoise"]
    original = denoise.read_text(encoding="utf-8")
    denoise.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ReportError, match="duplicate"):
        write_final_report(report_case.context)
    denoise.write_text(original, encoding="utf-8")
    manifest_path = report_case.context.qc_manifest_json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figure_metadata"]["input"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ReportError, match="metadata mismatch"):
        write_final_report(report_case.context)


def test_report_partial_commit_failure_removes_owned_outputs(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    real_write = qc_module.os.write
    calls = 0

    def short_then_fail(descriptor, content):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, content[:10])
        raise OSError("injected report short-write failure")

    monkeypatch.setattr(qc_module.os, "write", short_then_fail)
    with pytest.raises(ReportError, match="commit"):
        write_final_report(report_case.context)
    assert list(report_case.context.output_directory.iterdir()) == []


def test_report_render_temp_cleanup_preserves_reused_foreign_root(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._render_pdf
    state: dict[str, Path] = {}
    foreign_bytes = b"foreign report staging root\n"

    def rename_root_after_render(context, data, path):
        original(context, data, path)
        root = path.parent
        held = root.with_name(f"{root.name}.held-report-render")
        root.rename(held)
        (root / "nested").mkdir(parents=True, mode=0o700)
        foreign = root / "nested" / "foreign.bin"
        foreign.write_bytes(foreign_bytes)
        state.update(root=root, held=held, foreign=foreign)

    monkeypatch.setattr(report_module, "_render_pdf", rename_root_after_render)
    try:
        with pytest.raises(
            ReportError, match="temporary|staging|binding|changed"
        ):
            write_final_report(report_case.context)

        assert state["foreign"].read_bytes() == foreign_bytes
        assert not state["held"].exists() or list(state["held"].iterdir()) == []
        if state["held"].exists():
            assert not any(path.is_file() for path in state["held"].rglob("*"))
        assert list(report_case.context.output_directory.iterdir()) == []
    finally:
        for key in ("held", "root"):
            candidate = state.get(key)
            if candidate is not None and candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_report_render_writes_owned_fd_not_foreign_replacement(
    report_case: SyntheticReportCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._render_pdf
    state: dict[str, Path] = {}
    foreign_bytes = b"foreign PDF target must remain unchanged\n"

    def swap_before_render(context, data, target):
        root = target.parent
        held = root.with_name(f"{root.name}.held-before-report-render")
        root.rename(held)
        root.mkdir(mode=0o700)
        foreign = root / target.name
        foreign.write_bytes(foreign_bytes)
        state.update(root=root, held=held, foreign=foreign)
        original(context, data, target)

    monkeypatch.setattr(report_module, "_render_pdf", swap_before_render)
    try:
        with pytest.raises(
            ReportError,
            match="temporary|binding|changed",
        ):
            write_final_report(report_case.context)

        assert state["foreign"].read_bytes() == foreign_bytes
        assert not state["held"].exists()
        assert list(report_case.context.output_directory.iterdir()) == []
    finally:
        for key in ("held", "root"):
            candidate = state.get(key)
            if candidate is not None and candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_report_destination_swap_after_preflight_writes_no_foreign_files(
    report_case: SyntheticReportCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._render_pdf
    destination = report_case.context.output_directory
    held = destination.with_name(f"{destination.name}.held-report-output")

    def swap_destination_after_render(context, data, path):
        original(context, data, path)
        destination.rename(held)
        destination.mkdir(mode=0o700)

    monkeypatch.setattr(
        report_module,
        "_render_pdf",
        swap_destination_after_render,
    )
    try:
        with pytest.raises(
            ReportError,
            match="output directory.*replaced|binding",
        ):
            write_final_report(report_case.context)

        assert destination.exists() and list(destination.iterdir()) == []
        assert held.exists() and list(held.iterdir()) == []
    finally:
        for candidate in (held, destination):
            if candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_report_destination_swap_mid_copy_rolls_back_pinned_original(
    report_case: SyntheticReportCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = report_case.context.output_directory
    held = destination.with_name(f"{destination.name}.held-report-mid-copy")
    real_write_all = qc_module._write_all
    swapped = False

    def write_then_swap(descriptor, content):
        nonlocal swapped
        real_write_all(descriptor, content)
        if not swapped:
            swapped = True
            destination.rename(held)
            destination.mkdir(mode=0o700)

    monkeypatch.setattr(qc_module, "_write_all", write_then_swap)
    try:
        with pytest.raises(
            ReportError,
            match="output directory.*replaced|binding",
        ):
            write_final_report(report_case.context)

        assert swapped
        assert destination.exists() and list(destination.iterdir()) == []
        assert held.exists() and list(held.iterdir()) == []
    finally:
        for candidate in (held, destination):
            if candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_report_preserves_commit_cleanup_note_on_mapped_error(
    report_case: SyntheticReportCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module
    import dmri_pipeline.report as report_module

    real_commit = report_module._commit_files
    real_write_all = qc_module._write_all
    real_unlink = qc_module.os.unlink
    state: dict[str, int] = {}

    def track_destination(destination, sources, **kwargs):
        state["root_descriptor"] = destination.root_descriptor
        return real_commit(destination, sources, **kwargs)

    def write_then_fail(descriptor, content):
        real_write_all(descriptor, content)
        raise OSError("injected report commit primary failure")

    def fail_destination_unlink(path, *, dir_fd=None):
        if dir_fd == state.get("root_descriptor"):
            raise OSError("injected persistent report rollback failure")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(report_module, "_commit_files", track_destination)
    monkeypatch.setattr(qc_module, "_write_all", write_then_fail)
    monkeypatch.setattr(qc_module.os, "unlink", fail_destination_unlink)
    try:
        with pytest.raises(ReportError, match="commit") as captured:
            write_final_report(report_case.context)

        notes = getattr(captured.value, "__notes__", ())
        assert any("rollback cleanup" in note for note in notes)
        leftovers = list(report_case.context.output_directory.iterdir())
        assert leftovers
        assert all(path.stat().st_size == 0 for path in leftovers)
    finally:
        monkeypatch.setattr(qc_module.os, "unlink", real_unlink)
        for path in report_case.context.output_directory.iterdir():
            path.unlink()


def test_report_detects_modify_then_restore_and_external_hardlink(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._render_pdf

    def mutate_restore(context, data, path):
        original(context, data, path)
        exact = context.global_csv.read_bytes()
        context.global_csv.write_bytes(exact + b"x")
        context.global_csv.write_bytes(exact)

    monkeypatch.setattr(report_module, "_render_pdf", mutate_restore)
    with pytest.raises(ReportError, match="changed"):
        write_final_report(report_case.context)
    assert list(report_case.context.output_directory.iterdir()) == []

    external = report_case.context.global_csv.with_name("external_global_alias.csv")
    os.link(report_case.context.global_csv, external)
    with pytest.raises(ReportError, match="link count"):
        write_final_report(report_case.context)


def test_real_fsl_outlier_report_parser_and_fail_closed(tmp_path: Path) -> None:
    import dmri_pipeline.report as report_module

    real = tmp_path / "real_report.txt"
    real.write_text(
        "Slice 6 in scan 109 is an outlier with mean 4.1 standard deviations off\n"
        "Slice 7 in scan 109 is an outlier with mean 4.2 standard deviations off\n",
        encoding="utf-8",
    )
    assert report_module._reported_outlier_count(real) == 2
    empty = tmp_path / "empty_report.txt"
    empty.write_text("", encoding="utf-8")
    assert report_module._reported_outlier_count(empty) == 0
    unsupported = tmp_path / "unsupported_report.txt"
    unsupported.write_text("outlier report version unknown\n", encoding="utf-8")
    with pytest.raises(ReportError, match="unsupported"):
        report_module._reported_outlier_count(unsupported)


def test_markdown_stage_tables_are_contiguous_and_structurally_valid(
    report_case: SyntheticReportCase,
) -> None:
    markdown = write_final_report(report_case.context).markdown.read_text(
        encoding="utf-8"
    )
    provenance = markdown.split("## Ordered stage provenance\n\n", 1)[1]
    first_table, remainder = provenance.split("\n\n", 1)
    rows = first_table.splitlines()
    assert all(line.startswith("|") and line.endswith("|") for line in rows)
    assert len(rows) == 16  # header, separator, and 14 contiguous stage rows
    assert "## Stage hash and output detail" in remainder


def test_embedded_machine_paths_are_removed_from_all_report_payloads(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module
    from dmri_pipeline.state import StageContext

    stage = report_case.context.stage_context
    unsafe_stage = StageContext(
        stage.config,
        stage.package_root,
        stage.subject_root,
        {
            "fsl": "FSL 6.0 (/opt/fsl)",
            "matlab": r"MATLAB R2024a (C:\Applications\MATLAB)",
            "probe": "details at file://machine/private",
            "uri": "tool https://machine.invalid/path",
            "bracket": "FSL[/opt/fsl-bracket]",
            "semicolon": "FSL;/opt/fsl-semicolon",
            "unc": "tool //server/share/bin",
            "unicode": "tool [/用户/数据]",
        },
    )
    unsafe_context = replace(report_case.context, stage_context=unsafe_stage)
    observed: list[str] = []
    original = report_module._render_pdf

    def inspect_pdf_payload(context, data, path):
        observed.append(str(data.payload))
        original(context, data, path)

    monkeypatch.setattr(report_module, "_render_pdf", inspect_pdf_payload)
    outputs = write_final_report(unsafe_context)
    combined = (
        outputs.run_summary_json.read_text(encoding="utf-8")
        + outputs.markdown.read_text(encoding="utf-8")
        + "\n".join(observed)
    )
    for forbidden in (
        "/opt/fsl",
        r"C:\Applications",
        "file://",
        "https://",
        "//server/share",
        "/用户/数据",
    ):
        assert forbidden not in combined


def test_machine_path_detector_covers_punctuation_unc_and_unicode() -> None:
    import dmri_pipeline.report as report_module

    unsafe = (
        "FSL[/opt/fsl]",
        "FSL;/opt/fsl",
        r"MATLAB[C:\Applications\MATLAB]",
        "tool //server/share/bin",
        "tool [/用户/数据]",
    )
    for value in unsafe:
        assert report_module._sanitize_string(value) == "[path removed]"
        with pytest.raises(ReportError, match="private path"):
            report_module._validate_serialized_report(
                json.dumps({"software": value})
            )
    for safe in ("PA/AP", "../qc/figure.png", "ratio / value"):
        assert not report_module._contains_machine_path(safe)


def test_markdown_cells_and_pdf_summaries_are_bounded() -> None:
    import dmri_pipeline.report as report_module

    assert report_module._markdown_cell("a|b\nc`d") == (
        "a&#124;b<br>c&#96;d"
    )
    quad = {f"qc_motion_{index:02d}": float(index) for index in range(20)}
    quad_text = report_module._format_quad(quad)
    assert "14 additional metrics" in quad_text
    assert len(quad_text.splitlines()) == 8
    histogram = [
        {"error_code": index, "voxel_count": index + 1}
        for index in range(20)
    ]
    histogram_text = report_module._format_error_histogram(histogram)
    assert "18 additional codes" in histogram_text
    assert len(histogram_text.splitlines()) == 3


def test_pdf_page5_many_error_codes_stays_inside_summary_panel(
    report_case: SyntheticReportCase,
) -> None:
    import matplotlib.pyplot as plt
    import dmri_pipeline.report as report_module

    expected = report_module._expected_figure_paths(report_case.context)
    guarded = [
        *report_module._context_inputs(report_case.context),
        *expected.values(),
    ]
    with report_module._InputGuard(guarded) as snapshots:
        figures = report_module._validate_qc_manifest(
            report_case.context, snapshots, expected
        )
        data = report_module._collect_report_data(
            report_case.context, figures, snapshots
        )
        payload = json.loads(
            json.dumps(report_module._json_ready(data.payload))
        )
        payload["noddi"]["error_code_histogram"] = [
            {"error_code": index, "voxel_count": index + 1}
            for index in range(20)
        ]
        figure = report_module._make_pdf_page5(
            replace(data, payload=payload), "synthetic footer"
        )
        plt.close(figure)


def test_report_rejects_transient_snapshot_symlink_swap(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._render_pdf

    def transient_swap(context, data, path):
        source = data.figures["stripe"]
        snapshot = data.snapshots[source]
        held = snapshot.with_name(f"{snapshot.name}.held")
        os.chmod(snapshot.parent, 0o700)
        snapshot.rename(held)
        try:
            snapshot.symlink_to(source)
            original(context, data, path)
        finally:
            snapshot.unlink()
            held.rename(snapshot)
            os.chmod(snapshot.parent, 0o500)

    monkeypatch.setattr(report_module, "_render_pdf", transient_swap)
    with pytest.raises(ReportError, match="snapshot directory changed"):
        write_final_report(report_case.context)
    assert list(report_case.context.output_directory.iterdir()) == []


def test_report_cleanup_preserves_foreign_reused_snapshot_root_and_primary_error(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module

    state: dict[str, Path] = {}
    foreign_bytes = b"foreign report cleanup sentinel\n"

    def fail_after_root_reuse(context, data, path):
        snapshot = data.snapshots[data.figures["stripe"]]
        root = snapshot.parent
        held = root.with_name(f"{root.name}.held-report-test")
        os.chmod(root, 0o700)
        root.rename(held)
        (root / "nested").mkdir(parents=True, mode=0o700)
        foreign = root / "nested" / "foreign.bin"
        foreign.write_bytes(foreign_bytes)
        state.update(root=root, held=held, foreign=foreign)
        raise RuntimeError("primary render failure")

    monkeypatch.setattr(report_module, "_render_pdf", fail_after_root_reuse)
    try:
        with pytest.raises(RuntimeError, match="primary render failure"):
            write_final_report(report_case.context)

        assert state["foreign"].read_bytes() == foreign_bytes
        assert not state["held"].exists() or list(state["held"].iterdir()) == []
        if state["held"].exists():
            assert not any(path.is_file() for path in state["held"].rglob("*"))
        assert list(report_case.context.output_directory.iterdir()) == []
    finally:
        for key in ("held", "root"):
            candidate = state.get(key)
            if candidate is not None and candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_markdown_stress_is_structured_repr_free_and_bounded(
    report_case: SyntheticReportCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.report as report_module

    original = report_module._collect_report_data
    captured: list[dict[str, object]] = []
    quad = {
        f"qc_motion_metric_{index:02d}": index + 0.125
        for index in range(20)
    }
    histogram = [
        {"error_code": index, "voxel_count": index + 100}
        for index in range(20)
    ]

    def stress_data(context, figures, snapshots):
        data = original(context, figures, snapshots)
        payload = json.loads(
            json.dumps(report_module._json_ready(data.payload))
        )
        payload["eddy"]["eddy_quad_selected_numeric_metrics"] = quad
        payload["noddi"]["error_code_histogram"] = histogram
        captured.append(payload)
        return replace(data, payload=payload)

    monkeypatch.setattr(report_module, "_collect_report_data", stress_data)
    outputs = write_final_report(report_case.context)
    markdown = outputs.markdown.read_text(encoding="utf-8")
    payload = captured[0]

    assert "OrderedDict(" not in markdown
    assert "mappingproxy(" not in markdown
    assert "dict(" not in markdown
    assert "{" not in markdown and "}" not in markdown
    assert re.search(r"`\[[^\n]*\]`", markdown) is None
    assert max(len(line) for line in markdown.splitlines()) <= 360

    lines = markdown.splitlines()
    index = 0
    table_count = 0
    while index < len(lines):
        if not lines[index].startswith("|"):
            index += 1
            continue
        table: list[str] = []
        while index < len(lines) and lines[index].startswith("|"):
            table.append(lines[index])
            index += 1
        table_count += 1
        assert len(table) >= 2
        width = table[0].count("|")
        assert width >= 3
        assert all(row.endswith("|") and row.count("|") == width for row in table)
    assert table_count >= 10

    provenance = markdown.split("## Ordered stage provenance\n\n", 1)[1]
    provenance_table = provenance.split("\n\n", 1)[0].splitlines()
    assert len(provenance_table) == 16
    detail = markdown.split("## Stage hash and output detail\n\n", 1)[1]
    detail_table = detail.split("\n\n", 1)[0].splitlines()
    assert len(detail_table) == 16

    for key, value in quad.items():
        assert key in markdown
        assert f"{value:.12g}" in markdown
    for entry in histogram:
        assert f"| {entry['error_code']} | {entry['voxel_count']} |" in markdown
    for resource in payload["atlas"]["resource_sha256"].values():
        assert resource["sha256"] in markdown
    for field, count in payload["summary_mask"].items():
        assert field in markdown and f"| {count} |" in markdown
    for model_name in payload["models"]:
        assert model_name in markdown

    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown):
        assert "://" not in link and not Path(link).is_absolute()
        assert (report_case.context.output_directory / link).resolve().exists()


def test_pdf_summary_builders_are_structured_and_repr_free(
    report_case: SyntheticReportCase,
) -> None:
    import dmri_pipeline.report as report_module

    figures = report_module._expected_figure_paths(report_case.context)
    guarded = [
        *report_module._context_inputs(report_case.context),
        *figures.values(),
    ]
    with report_module._InputGuard(guarded) as snapshots:
        figures = report_module._validate_qc_manifest(
            report_case.context, snapshots, figures
        )
        data = report_module._collect_report_data(
            report_case.context, figures, snapshots
        )
        summaries = (
            *report_module._pdf_page4_summary(data.payload),
            *report_module._pdf_page5_summary(data.payload),
        )
    text = "\n".join(summaries)
    assert "OrderedDict" not in text
    assert "mappingproxy" not in text
    assert "[{" not in text


def test_pdf_stage_provenance_contains_all_full_hash_evidence(
    report_case: SyntheticReportCase,
) -> None:
    import dmri_pipeline.report as report_module

    expected = report_module._expected_figure_paths(report_case.context)
    guarded = [
        *report_module._context_inputs(report_case.context),
        *expected.values(),
    ]
    with report_module._InputGuard(guarded) as snapshots:
        figures = report_module._validate_qc_manifest(
            report_case.context, snapshots, expected
        )
        data = report_module._collect_report_data(
            report_case.context, figures, snapshots
        )
        blocks = report_module._pdf_page6_provenance(data.payload)
    assert len(blocks) == 14
    for block, stage in zip(blocks, data.payload["stages"], strict=True):
        assert stage["stage"] in block
        assert stage["started_utc"] in block
        assert stage["completed_utc"] in block
        assert stage["stage_signature"] in block
        assert report_module._stage_output_hash_aggregate(stage) in block
