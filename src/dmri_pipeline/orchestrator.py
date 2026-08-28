"""One-subject orchestration for the reproducible dMRI pipeline."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

import nibabel as nib
import numpy as np

from .audit import InputAudit, audit_inputs, write_input_audit
from .config import PipelineConfig
from .fsl import (
    ExternalCommandError,
    FSLContext,
    FSLInstallation,
    build_bet_command,
    build_eddy_command,
    build_eddy_quad_command,
    build_jhu_commands,
    build_topup_command,
    build_topup_mean_command,
    discover_fsl,
    run_fsl_command,
)
from .models import ModelContext, fit_direct_dki, fit_dki, fit_dti
from .noddi import (
    MATLABInstallation,
    NODDIContext,
    NODDIError,
    NODDIExternalCommandError,
    build_merge_command,
    build_prepare_command,
    build_worker_command,
    choose_noddi_workers,
    discover_matlab,
    launch_noddi_workers,
    merge_noddi,
    prepare_noddi,
)
from .preprocess import (
    PreprocessContext,
    PreprocessError,
    clean_bet_mask,
    prepare_topup_inputs,
    run_denoise,
    run_gibbs,
)
from .qc import (
    DKI_DIRECT_KEYS,
    DKI_KEYS,
    DTI_KEYS,
    FIGURE_FILENAMES,
    NODDI_KEYS,
    StageQCContext,
    generate_all_qc,
)
from .report import REPORT_STAGE_ORDER, ReportContext, write_final_report
from .resources import validate_jhu_resource
from .state import (
    StageContext,
    StageOutcome,
    StageRecord,
    StageRunner,
    StageSpec,
    StageStateError,
)
from .stripe_qc import (
    QCDecision,
    expected_stripe_detail_paths,
    run_stripe_qc,
)
from .summary import CANONICAL_METRICS, SummaryContext, summarize_subject
from .utils import InputAuditError, normalize_bvecs, round_shells


STAGE_ORDER = (
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
    "report",
)

_FSL_STAGES = frozenset(
    {"01_denoise", "03_topup", "04_bet", "05_eddy", "09_jhu_48roi"}
)
_MATLAB_STAGES = frozenset({"08_noddi"})
_CONTINUE_QC = {"INCLUDE", "INCLUDE_WITH_FLAGS", "INCLUDE_AFTER_REVIEW"}
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = Path(__file__).resolve().parent
_ATLAS_DIR = _PACKAGE_ROOT / "resources" / "jhu_48roi"
_ATLAS_IMAGE = _ATLAS_DIR / "JHU-ICBM-labels-2mm.nii.gz"
_ATLAS_XML = _ATLAS_DIR / "JHU-labels.xml"
_ATLAS_PROVENANCE = _ATLAS_DIR / "provenance.json"
_HENRIQUE_HELPER = _PACKAGE_ROOT / "vendor" / "henrique_helpers" / "dki_alternative.py"
_EDDY_QUAD_SCHEMA = frozenset(
    (
        "eddy_input_flag",
        "eddy_input",
        "data_file_eddy",
        "data_file_mask",
        "data_file_bvals",
        "data_no_dw_vols",
        "data_no_b0_vols",
        "data_no_PE_dirs",
        "data_protocol",
        "data_no_shells",
        "data_unique_bvals",
        "data_unique_pes",
        "data_eddy_para",
        "data_vox_size",
        "qc_path",
        "qc_mot_abs",
        "qc_mot_rel",
        "qc_params_flag",
        "qc_params_avg",
        "qc_s2v_params_flag",
        "qc_s2v_params_avg_std",
        "qc_field_flag",
        "qc_vox_displ_std",
        "qc_ol_flag",
        "qc_outliers_tot",
        "qc_outliers_b",
        "qc_outliers_pe",
        "qc_cnr_flag",
        "qc_cnr_avg",
        "qc_cnr_std",
        "qc_rss_flag",
    )
)
_EDDY_QUAD_NO_OUTLIER_SOURCES = frozenset(
    (
        "not-emitted-by-eddy-quad",
        "eddy-quad-legacy-residual-msr",
    )
)
_MAX_EDDY_QUAD_PDF_BYTES = 128 * 1024 * 1024
_MAX_EDDY_QUAD_TEXT_BYTES = 64 * 1024 * 1024


class PipelineExternalError(RuntimeError):
    """An invoked scientific command failed to launch or exited nonzero."""


class PipelineDependencyError(RuntimeError):
    """A required package/software dependency is missing or unsafe."""


class PipelineInputError(ValueError):
    """Configuration places protected inputs/resources in an unsafe identity."""


class PipelineOutputError(RuntimeError):
    """A stage produced scientifically invalid or unsafe output."""


@dataclass(frozen=True)
class PipelineStageOutcome:
    """One ordered orchestration result."""

    stage: str
    status: str
    directory: Path
    record_path: Path | None = None


@dataclass(frozen=True)
class PipelineOutcome:
    """Immutable outcome of one validation, plan, or mutating invocation."""

    subject: str
    status: str
    stages: tuple[PipelineStageOutcome, ...]
    subject_output: Path

    @property
    def stage_statuses(self) -> tuple[tuple[str, str], ...]:
        return tuple((stage.stage, stage.status) for stage in self.stages)


@dataclass(frozen=True)
class StageSelection:
    """One optional bounded execution request over the fixed stage order."""

    stop_after: str | None = None
    only_stage: str | None = None

    def __post_init__(self) -> None:
        if self.stop_after is not None and self.only_stage is not None:
            raise ValueError("stop_after and only_stage are mutually exclusive")
        for label, value in (
            ("stop_after", self.stop_after),
            ("only_stage", self.only_stage),
        ):
            if value is not None and value not in STAGE_ORDER:
                raise ValueError(f"unknown {label} stage: {value}")

    @property
    def execution_names(self) -> tuple[str, ...]:
        if self.only_stage is not None:
            return (self.only_stage,)
        if self.stop_after is not None:
            return STAGE_ORDER[: STAGE_ORDER.index(self.stop_after) + 1]
        return STAGE_ORDER

    @property
    def required_names(self) -> tuple[str, ...]:
        if self.only_stage is not None:
            return STAGE_ORDER[: STAGE_ORDER.index(self.only_stage) + 1]
        return self.execution_names

    @property
    def success_status(self) -> str:
        if self.only_stage is not None:
            return "STAGE_COMPLETE"
        if self.stop_after is not None:
            return "PARTIAL_COMPLETE"
        return "COMPLETE"


def _validate_selection(
    mode: str,
    force_stage: str | None,
    selection: StageSelection,
) -> None:
    if not isinstance(selection, StageSelection):
        raise TypeError("selection must be a StageSelection")
    if mode != "run" and selection != StageSelection():
        raise ValueError("bounded execution is valid only for a normal run")
    if force_stage is not None and force_stage not in selection.execution_names:
        raise ValueError(
            f"forced stage {force_stage} is outside the selected execution range"
        )


@dataclass
class _Runtime:
    config: PipelineConfig
    fsl: FSLInstallation | None = None
    matlab: MATLABInstallation | None = None

    def require_fsl(self) -> FSLInstallation:
        if self.fsl is None:
            self.fsl = discover_fsl(self.config)
        return self.fsl

    def require_matlab(self) -> MATLABInstallation:
        if self.matlab is None:
            self.matlab = discover_matlab(self.config)
        return self.matlab


class _SubjectLock:
    """Exclusive, nonblocking lock on one real subject directory."""

    def __init__(self, subject_root: Path) -> None:
        self.subject_root = Path(subject_root)
        self.path = self.subject_root / ".pipeline.lock"
        self._descriptor: int | None = None
        self._directory_descriptor: int | None = None
        self._anchor_descriptor: int | None = None
        self._anchor_directory_descriptor: int | None = None

    def __enter__(self) -> "_SubjectLock":
        anchor_directory, anchor = _acquire_subject_lock_anchor(self.subject_root)
        self._anchor_directory_descriptor = anchor_directory
        self._anchor_descriptor = anchor
        try:
            _require_safe_subject_root(self.subject_root)
            directory_descriptor = _open_directory_chain(self.subject_root)
            directory_metadata = os.fstat(directory_descriptor)
            if not _directory_path_matches(self.subject_root, directory_metadata):
                os.close(directory_descriptor)
                raise StageStateError(
                    "subject output identity changed while pinning its lock directory"
                )
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            try:
                descriptor = os.open(
                    ".pipeline.lock", flags, 0o600, dir_fd=directory_descriptor
                )
            except OSError as error:
                os.close(directory_descriptor)
                raise StageStateError(
                    f"cannot open safe subject lock: {self.path}"
                ) from error
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise StageStateError(
                        "subject lock must be a single-link regular file"
                    )
                named = os.stat(
                    ".pipeline.lock",
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(named.st_mode)
                    or (named.st_dev, named.st_ino)
                    != (metadata.st_dev, metadata.st_ino)
                ):
                    raise StageStateError(
                        "subject lock identity changed during open"
                    )
                if not _directory_path_matches(
                    self.subject_root, directory_metadata
                ):
                    raise StageStateError(
                        "subject output identity changed while opening its lock"
                    )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise StageStateError(
                        f"a pipeline is already running for subject "
                        f"{self.subject_root.name}"
                    ) from error
                if not _directory_path_matches(
                    self.subject_root, directory_metadata
                ):
                    raise StageStateError(
                        "subject output identity changed while acquiring its lock"
                    )
            except BaseException:
                os.close(descriptor)
                os.close(directory_descriptor)
                raise
            self._descriptor = descriptor
            self._directory_descriptor = directory_descriptor
            return self
        except BaseException:
            _unlock_close(self._anchor_descriptor)
            self._anchor_descriptor = None
            if self._anchor_directory_descriptor is not None:
                os.close(self._anchor_directory_descriptor)
                self._anchor_directory_descriptor = None
            raise

    def __exit__(self, *_args: object) -> None:
        _unlock_close(self._descriptor)
        self._descriptor = None
        if self._directory_descriptor is not None:
            os.close(self._directory_descriptor)
            self._directory_descriptor = None
        _unlock_close(self._anchor_descriptor)
        self._anchor_descriptor = None
        if self._anchor_directory_descriptor is not None:
            os.close(self._anchor_directory_descriptor)
            self._anchor_directory_descriptor = None


def build_plan(config: PipelineConfig) -> list[StageSpec]:
    """Return the real ordered stage specifications without executing them."""
    _require_config(config)
    return _build_plan(config, _Runtime(config))


def run_pipeline(
    config: PipelineConfig,
    mode: str,
    force_stage: str | None = None,
    *,
    selection: StageSelection | None = None,
) -> PipelineOutcome:
    """Validate, describe, or execute one subject pipeline."""
    _require_config(config)
    if mode not in {"run", "validate-only", "dry-run"}:
        raise ValueError("mode must be 'run', 'validate-only', or 'dry-run'")
    if force_stage is not None and force_stage not in STAGE_ORDER:
        raise ValueError(f"unknown force stage: {force_stage}")
    if mode != "run" and force_stage is not None:
        raise ValueError("--force-stage is valid only for a normal run")
    selection = StageSelection() if selection is None else selection
    _validate_selection(mode, force_stage, selection)

    _validate_subject_input_separation(config)
    audit = audit_inputs(config)
    validate_jhu_resource(_ATLAS_IMAGE, _ATLAS_XML)
    if mode != "run":
        runtime = _discover_runtime(config)
        plan = _build_plan(config, runtime)
        _validate_plan_sources(plan)
        if mode == "validate-only":
            print(
                f"VALIDATED subject={config.subject_id} "
                f"output={config.subject_output}"
            )
            return PipelineOutcome(
                config.subject_id, "VALIDATED", (), config.subject_output
            )
        stages = _dry_run(config, audit, plan, runtime)
        return PipelineOutcome(
            config.subject_id, "DRY_RUN", stages, config.subject_output
        )

    runtime = _Runtime(config)
    plan = _build_plan(config, runtime)
    _validate_plan_sources(plan)
    _create_subject_root(config.subject_output)
    gate_runner = _stage_runner(config, runtime, STAGE_ORDER[0])
    outcomes: list[PipelineStageOutcome] = []
    with _SubjectLock(config.subject_output):
        if force_stage is not None:
            gate_runner.invalidate_from(STAGE_ORDER, force_stage)
        elif _safe_manual_review_transition(config, gate_runner, audit):
            gate_runner.invalidate_from(STAGE_ORDER, STAGE_ORDER[0])

        gate_plan = plan[:2]
        _reject_unsafe_existing_state(gate_runner, gate_plan)
        for spec in gate_plan:
            runner = _stage_runner(config, runtime, spec.name)
            print(f"[{spec.name}] checking")
            result = runner.run(spec)
            outcome = _pipeline_stage_outcome(result)
            outcomes.append(outcome)
            print(f"[{spec.name}] {outcome.status}")
            if spec.name == "00_pre_denoise_motion_qc":
                decision = _read_qc_decision(result.directory)
                if decision.status == "EXCLUDE":
                    return PipelineOutcome(
                        config.subject_id,
                        "EXCLUDED",
                        tuple(outcomes),
                        config.subject_output,
                    )
                if decision.status == "HOLD_FOR_REVIEW":
                    return PipelineOutcome(
                        config.subject_id,
                        "HOLD_FOR_REVIEW",
                        tuple(outcomes),
                        config.subject_output,
                    )
                if decision.status not in _CONTINUE_QC:
                    raise StageStateError(
                        f"unsupported stripe-QC decision: {decision.status}"
                    )
        runtime.require_fsl()
        runtime.require_matlab()
        plan = _build_plan(config, runtime)
        _validate_plan_sources(plan)
        scientific_plan = plan[2:]
        stage_runs = tuple(
            (spec, _stage_runner(config, runtime, spec.name))
            for spec in scientific_plan
        )
        for spec, runner in stage_runs:
            _reject_unsafe_existing_state(runner, (spec,))
        for spec, runner in stage_runs:
            print(f"[{spec.name}] checking")
            result = runner.run(spec)
            outcome = _pipeline_stage_outcome(result)
            outcomes.append(outcome)
            print(f"[{spec.name}] {outcome.status}")
    return PipelineOutcome(
        config.subject_id, "COMPLETE", tuple(outcomes), config.subject_output
    )


def _build_plan(config: PipelineConfig, runtime: _Runtime) -> list[StageSpec]:
    root = config.subject_output
    paths = _paths(config)
    qc_software = _stage_software_provenance(runtime, "qc")
    report_software = _stage_software_provenance(runtime, "report")
    stripe_detail_paths = expected_stripe_detail_paths(
        config, root / "00_pre_denoise_motion_qc"
    )

    def audit_action(work: Path) -> None:
        result = audit_inputs(config)
        write_input_audit(result, work / "input_audit.json")
        _write_canonical_bvecs(
            config.bvecs, work / "bvecs_fsl_3xN", int(result.pa_shape[3])
        )

    def audit_validator(work: Path) -> Sequence[Path]:
        result = audit_inputs(config)
        record = _read_json(work / "input_audit.json", "input audit")
        if record != result.to_dict():
            raise StageStateError("input audit record does not match fresh read-only audit")
        canonical = _load_text(work / "bvecs_fsl_3xN", "canonical b-vectors")
        if canonical.shape != (3, int(result.pa_shape[3])) or not np.isfinite(canonical).all():
            raise StageStateError("canonical b-vectors must be finite 3xN")
        return (work / "input_audit.json", work / "bvecs_fsl_3xN")

    def stripe_action(work: Path) -> None:
        run_stripe_qc(config, work)

    def stripe_validator(work: Path) -> Sequence[Path]:
        _read_qc_decision(work)
        files = _regular_tree_files(work)
        required = {
            "stripe_metrics.csv",
            "stripe_decision.json",
            "automatic_summary.txt",
            "00_raw_b0_anatomy_overview.png",
            "01_cSI_by_volume.png",
            "02_cSI_by_shell.png",
        }
        if not required.issubset(path.name for path in files):
            raise StageStateError("stripe-QC output set is incomplete")
        actual_details = {
            path
            for path in files
            if path.name.startswith("03_candidate_details_")
            or path.name.startswith("04_all_volumes_")
        }
        expected_details = {
            work / detail.name for detail in stripe_detail_paths
        }
        if actual_details != expected_details:
            raise StageStateError(
                "stripe-QC detail sheets do not match the deterministic "
                "candidate/all-volume contract"
            )
        return files

    def denoise_action(work: Path) -> None:
        installation = runtime.require_fsl()

        def execute(argv: Sequence[str]):
            return run_fsl_command(
                argv, work / "denoise_fsl.log", installation.environment
            )

        try:
            run_denoise(
                _preprocess_context(
                    config,
                    audit_inputs(config),
                    denoise_dir=work,
                    command_executor=execute,
                    fsldir=installation.fsldir,
                )
            )
        except PreprocessError as error:
            _raise_preprocess_boundary(error)

    def gibbs_action(work: Path) -> None:
        try:
            run_gibbs(
                _preprocess_context(
                    config, audit_inputs(config), gibbs_dir=work
                )
            )
        except PreprocessError as error:
            raise PipelineOutputError(str(error)) from error

    def topup_action(work: Path) -> None:
        installation = runtime.require_fsl()
        audit_value = audit_inputs(config)
        try:
            prepare_topup_inputs(
                _preprocess_context(config, audit_value, topup_dir=work)
            )
        except PreprocessError as error:
            raise PipelineOutputError(str(error)) from error
        fsl_context = _fsl_context(
            config, installation, audit_value, topup_dir=work
        )
        log = work / "topup_fsl.log"
        run_fsl_command(build_topup_command(fsl_context), log, installation.environment)
        _write_topup_metrics(fsl_context, work / "topup_metrics.json")

    def bet_action(work: Path) -> None:
        installation = runtime.require_fsl()
        audit_value = audit_inputs(config)
        fsl_context = _fsl_context(
            config, installation, audit_value, bet_dir=work
        )
        log = work / "bet_fsl.log"
        run_fsl_command(
            build_topup_mean_command(fsl_context), log, installation.environment
        )
        run_fsl_command(build_bet_command(fsl_context), log, installation.environment)
        try:
            clean_bet_mask(
                _preprocess_context(
                    config,
                    audit_value,
                    bet_dir=work,
                    bet_mask_source=Path(
                        f"{fsl_context.brain_prefix}_mask.nii.gz"
                    ),
                )
            )
        except PreprocessError as error:
            raise PipelineOutputError(str(error)) from error

    def eddy_action(work: Path) -> None:
        installation = runtime.require_fsl()
        audit_value = audit_inputs(config)
        fsl_context = _fsl_context(
            config, installation, audit_value, eddy_dir=work
        )
        log = work / "eddy_fsl.log"
        run_fsl_command(build_eddy_command(fsl_context), log, installation.environment)
        run_fsl_command(
            build_eddy_quad_command(fsl_context), log, installation.environment
        )
        bvals = _load_text(config.bvals, "b-values").reshape(-1)
        outliers = _load_eddy_outlier_map(
            Path(f"{fsl_context.eddy_prefix}.eddy_outlier_map"),
            int(audit_value.pa_shape[3]),
        )
        _write_eddy_quad_json(
            fsl_context.eddy_quad_output,
            work / "eddy_quad.json",
            expected_volume_count=int(audit_value.pa_shape[3]),
            expected_slices=int(
                audit_value.pa_shape[config.acquisition.slice_axis]
            ),
            bvals=bvals,
            outliers=outliers,
            expected_paths={
                "data_file_eddy": Path(f"{fsl_context.eddy_prefix}.nii.gz"),
                "data_file_mask": fsl_context.cleaned_mask,
                "data_file_bvals": config.bvals,
                "qc_path": fsl_context.eddy_quad_output,
            },
        )

    def dti_action(work: Path) -> None:
        fit_dti(_model_context(config, work))

    def dki_action(work: Path) -> None:
        fit_dki(_model_context(config, work))

    def direct_action(work: Path) -> None:
        fit_direct_dki(_model_context(config, work))

    def noddi_action(work: Path) -> None:
        matlab = runtime.require_matlab()
        workers = _pipeline_noddi_workers(config)
        context = NODDIContext(
            config=config,
            package_root=_PACKAGE_ROOT,
            stage_dir=work,
            eddy_dwi=paths["eddy_dwi"],
            cleaned_mask=paths["brain_mask"],
            bvals=config.bvals,
            rotated_bvecs=paths["rotated_bvecs"],
            matlab=matlab,
            workers=workers,
        )
        try:
            prepare_noddi(context)
            launch_noddi_workers(context)
            merge_noddi(context)
        except NODDIError as error:
            message = str(error)
            if "--force-stage" in message:
                message = message.replace("--force-stage 08_noddi ", "")
            error_type = (
                PipelineExternalError
                if isinstance(error, NODDIExternalCommandError)
                else PipelineOutputError
            )
            raise error_type(
                f"{message}; rerun ./run_pipeline.sh {config.config_path} "
                "to resume preserved checkpoints"
            ) from error

    def atlas_action(work: Path) -> None:
        validate_jhu_resource(_ATLAS_IMAGE, _ATLAS_XML)
        installation = runtime.require_fsl()
        fsl_context = _fsl_context(
            config, installation, audit_inputs(config), jhu_dir=work
        )
        log = work / "jhu_registration_fsl.log"
        for argv in build_jhu_commands(fsl_context):
            run_fsl_command(argv, log, installation.environment)
        _validate_subject_atlas(fsl_context.subject_atlas, paths["dti_fa"])

    def summary_action(work: Path) -> None:
        summarize_subject(
            SummaryContext(
                config=config,
                warped_atlas=paths["subject_atlas"],
                brain_mask=paths["brain_mask"],
                metric_maps=_summary_metric_maps(paths),
                noddi_error_codes=paths["noddi_error"],
                atlas_xml=_ATLAS_XML,
                atlas_provenance=_ATLAS_PROVENANCE,
                output_directory=work,
            )
        )

    def qc_action(work: Path) -> None:
        generate_all_qc(_qc_context(config, work, paths, qc_software))

    def report_action(work: Path) -> None:
        write_final_report(
            _report_context(
                config,
                work,
                paths,
                report_software,
                stripe_detail_paths=stripe_detail_paths,
            )
        )

    module = lambda name: _SOURCE_ROOT / f"{name}.py"
    common = (module("orchestrator"),)
    noddi_sources = tuple(
        sorted(
            (
                module("noddi"),
                *(_PACKAGE_ROOT / "scripts" / "matlab").glob("*.m"),
                *(
                    path
                    for root_dir in (
                        _PACKAGE_ROOT / "vendor" / "noddi_toolbox_v1.05",
                        _PACKAGE_ROOT / "vendor" / "nifti_matlab",
                    )
                    for path in root_dir.rglob("*")
                    if path.is_file()
                ),
            )
        )
    )
    atlas_resources = (_ATLAS_IMAGE, _ATLAS_XML, _ATLAS_PROVENANCE)
    stage_specs = [
        StageSpec(
            STAGE_ORDER[0],
            audit_action,
            audit_validator,
            (config.dwi_pa, config.bvals, config.bvecs, config.b0_ap),
            (*common, module("audit"), module("utils"), module("config")),
        ),
        StageSpec(
            STAGE_ORDER[1],
            stripe_action,
            stripe_validator,
            (config.dwi_pa, config.bvals, paths["audit_json"]),
            (*common, module("stripe_qc"), module("utils"), module("config")),
        ),
        StageSpec(
            STAGE_ORDER[2],
            denoise_action,
            lambda work: _required_outputs(
                work,
                (
                    "denoised_PA.nii.gz",
                    "denoised_AP.nii.gz",
                    "sigma_PA.nii.gz",
                    "sigma_AP.nii.gz",
                    "raw_mean_b0.nii.gz",
                    "raw_mean_b0_bet_mask.nii.gz",
                    "raw_mean_b0_bet_mask_dilated.nii.gz",
                    "denoise_metrics.json",
                    "denoise_fsl.log",
                ),
            ),
            (
                config.dwi_pa,
                config.b0_ap,
                config.bvals,
                paths["audit_json"],
                paths["stripe_decision"],
            ),
            (*common, module("preprocess"), module("audit"), module("fsl")),
        ),
        StageSpec(
            STAGE_ORDER[3],
            gibbs_action,
            lambda work: _required_outputs(
                work, ("gibbs_PA.nii.gz", "gibbs_AP.nii.gz", "gibbs_metrics.json")
            ),
            (
                paths["denoised_pa"],
                paths["denoised_ap"],
                paths["audit_json"],
            ),
            (*common, module("preprocess"), module("audit")),
        ),
        StageSpec(
            STAGE_ORDER[4],
            topup_action,
            lambda work: _validate_topup_outputs(
                work, audit_inputs(config), config
            ),
            (
                paths["gibbs_pa"],
                paths["gibbs_ap"],
                config.bvals,
                paths["audit_json"],
            ),
            (*common, module("preprocess"), module("fsl"), module("audit")),
        ),
        StageSpec(
            STAGE_ORDER[5],
            bet_action,
            _validate_bet_outputs,
            (
                paths["topup_corrected"],
                paths["topup_fieldcoef"],
                paths["topup_movpar"],
            ),
            (*common, module("preprocess"), module("fsl")),
        ),
        StageSpec(
            STAGE_ORDER[6],
            eddy_action,
            lambda work: _validate_eddy_outputs(
                work,
                audit_inputs(config),
                config.bvals,
                slice_axis=config.acquisition.slice_axis,
                expected_mask_path=paths["brain_mask"],
            ),
            (
                paths["gibbs_pa"],
                paths["brain_mask"],
                paths["acqparams_eddy"],
                paths["index_eddy"],
                config.bvals,
                paths["canonical_bvecs"],
                paths["topup_fieldcoef"],
                paths["topup_movpar"],
            ),
            (*common, module("fsl")),
        ),
        StageSpec(
            STAGE_ORDER[7],
            dti_action,
            lambda work: _required_outputs(
                work,
                ("FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz", "V1.nii.gz", "dti_metrics.json"),
            ),
            (
                paths["eddy_dwi"],
                paths["brain_mask"],
                config.bvals,
                paths["rotated_bvecs"],
            ),
            (*common, module("models"), module("utils")),
            (_HENRIQUE_HELPER,),
        ),
        StageSpec(
            STAGE_ORDER[8],
            dki_action,
            lambda work: _required_outputs(
                work,
                (
                    "FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz",
                    "V1.nii.gz", "MK.nii.gz", "AK.nii.gz", "RK.nii.gz",
                    "dki_metrics.json",
                ),
            ),
            (
                paths["eddy_dwi"],
                paths["brain_mask"],
                config.bvals,
                paths["rotated_bvecs"],
            ),
            (*common, module("models"), module("utils")),
            (_HENRIQUE_HELPER,),
        ),
        StageSpec(
            STAGE_ORDER[9],
            direct_action,
            lambda work: _required_outputs(
                work,
                ("MD.nii.gz", "MK.nii.gz", "S0.nii.gz", "dki_direct_metrics.json"),
            ),
            (
                paths["eddy_dwi"],
                paths["brain_mask"],
                config.bvals,
                paths["rotated_bvecs"],
            ),
            (*common, module("models"), module("utils")),
            (_HENRIQUE_HELPER,),
        ),
        StageSpec(
            STAGE_ORDER[10],
            noddi_action,
            _validate_noddi_outputs,
            (
                paths["eddy_dwi"],
                paths["brain_mask"],
                config.bvals,
                paths["rotated_bvecs"],
            ),
            (*common, *noddi_sources),
        ),
        StageSpec(
            STAGE_ORDER[11],
            atlas_action,
            _validate_jhu_outputs,
            (paths["dti_fa"],),
            (*common, module("fsl"), module("resources")),
            atlas_resources,
        ),
        StageSpec(
            STAGE_ORDER[12],
            summary_action,
            _summary_validator(config),
            tuple(
                (
                    paths["subject_atlas"],
                    paths["brain_mask"],
                    paths["noddi_error"],
                    *(_summary_metric_maps(paths).values()),
                )
            ),
            (*common, module("summary"), module("resources")),
            (_ATLAS_XML, _ATLAS_PROVENANCE),
        ),
        StageSpec(
            STAGE_ORDER[13],
            qc_action,
            _qc_validator(config),
            _qc_input_paths(config, paths),
            (*common, module("qc"), module("state"), module("utils")),
        ),
        StageSpec(
            STAGE_ORDER[14],
            report_action,
            _report_validator(config),
            _report_input_paths(
                config,
                paths,
                stripe_detail_paths=stripe_detail_paths,
            ),
            (*common, module("report"), module("qc"), module("summary"), module("state")),
            (_ATLAS_PROVENANCE,),
        ),
    ]
    if tuple(stage.name for stage in stage_specs) != STAGE_ORDER:
        raise AssertionError("internal stage order changed")
    return stage_specs


def _paths(config: PipelineConfig) -> dict[str, Path]:
    root = config.subject_output
    eddy_prefix = root / "05_eddy" / "eddy_unwarped_images"
    subject = config.subject_id
    return {
        "audit_json": root / "00_input_audit" / "input_audit.json",
        "canonical_bvecs": root / "00_input_audit" / "bvecs_fsl_3xN",
        "stripe_csv": root / "00_pre_denoise_motion_qc" / "stripe_metrics.csv",
        "stripe_decision": root / "00_pre_denoise_motion_qc" / "stripe_decision.json",
        "denoised_pa": root / "01_denoise" / "denoised_PA.nii.gz",
        "denoised_ap": root / "01_denoise" / "denoised_AP.nii.gz",
        "denoise_metrics": root / "01_denoise" / "denoise_metrics.json",
        "gibbs_pa": root / "02_gibbs" / "gibbs_PA.nii.gz",
        "gibbs_ap": root / "02_gibbs" / "gibbs_AP.nii.gz",
        "gibbs_metrics": root / "02_gibbs" / "gibbs_metrics.json",
        "merged_b0": root / "03_topup" / "PA_AP_b0.nii.gz",
        "topup_corrected": root / "03_topup" / "topup_corrected_b0s.nii.gz",
        "topup_field": root / "03_topup" / "topup_field_Hz.nii.gz",
        "topup_fieldcoef": root / "03_topup" / "topup_PA_AP_b0_fieldcoef.nii.gz",
        "topup_movpar": root / "03_topup" / "topup_PA_AP_b0_movpar.txt",
        "topup_manifest": root / "03_topup" / "topup_input_manifest.json",
        "topup_metrics": root / "03_topup" / "topup_metrics.json",
        "acqparams_eddy": root / "03_topup" / "acqparams_eddy.txt",
        "index_eddy": root / "03_topup" / "index_eddy.txt",
        "hifi_b0": root / "04_bet" / "hifi_nodif.nii.gz",
        "brain_mask": root / "04_bet" / "nodif_brain_mask.nii.gz",
        "bet_metrics": root / "04_bet" / "mask_cleanup_metrics.json",
        "eddy_dwi": Path(f"{eddy_prefix}.nii.gz"),
        "rotated_bvecs": Path(f"{eddy_prefix}.eddy_rotated_bvecs"),
        "eddy_parameters": Path(f"{eddy_prefix}.eddy_parameters"),
        "eddy_rms": Path(f"{eddy_prefix}.eddy_movement_rms"),
        "eddy_outlier_map": Path(f"{eddy_prefix}.eddy_outlier_map"),
        "eddy_outlier_report": Path(f"{eddy_prefix}.eddy_outlier_report"),
        "eddy_residuals": Path(f"{eddy_prefix}.eddy_residuals.nii.gz"),
        "eddy_cnr": Path(f"{eddy_prefix}.eddy_cnr_maps.nii.gz"),
        "eddy_quad_json": root / "05_eddy" / "eddy_quad.json",
        "dti_fa": root / "06_dti" / "FA.nii.gz",
        "dti_metrics": root / "06_dti" / "dti_metrics.json",
        "dki_metrics": root / "07_dki" / "dki_metrics.json",
        "direct_metrics": root / "07_dki_direct" / "dki_direct_metrics.json",
        "noddi_error": root / "08_noddi" / "NODDI_error_code.nii",
        "noddi_metrics": root / "08_noddi" / "noddi_metrics.json",
        "subject_atlas": root / "09_jhu_48roi" / "WM_JHU_ROIs.nii.gz",
        "summary_roi": root / "10_summary" / f"{subject}_JHU_48ROI_metrics.csv",
        "summary_global": root / "10_summary" / f"{subject}_global_metrics.csv",
        "summary_json": root / "10_summary" / f"{subject}_summary.json",
        "qc_manifest": root / "qc" / "qc_manifest.json",
    }


def _preprocess_context(
    config: PipelineConfig,
    audit: InputAudit,
    *,
    denoise_dir: Path | None = None,
    gibbs_dir: Path | None = None,
    topup_dir: Path | None = None,
    bet_dir: Path | None = None,
    bet_mask_source: Path | None = None,
    command_executor: Callable | None = None,
    fsldir: Path | None = None,
) -> PreprocessContext:
    root = config.subject_output
    effective = (
        config
        if fsldir is None
        else PipelineConfig(
            **{**config.__dict__, "fsldir": fsldir}
        )
    )
    kwargs: dict[str, object] = {}
    if command_executor is not None:
        kwargs["command_executor"] = command_executor
    return PreprocessContext(
        config=effective,
        audit=audit,
        denoise_dir=denoise_dir or root / "01_denoise",
        gibbs_dir=gibbs_dir or root / "02_gibbs",
        topup_dir=topup_dir or root / "03_topup",
        bet_dir=bet_dir or root / "04_bet",
        bet_mask_source=bet_mask_source
        or root / "04_bet" / "hifi_nodif_brain_mask.nii.gz",
        **kwargs,
    )


def _fsl_context(
    config: PipelineConfig,
    installation: FSLInstallation,
    audit: InputAudit,
    *,
    topup_dir: Path | None = None,
    bet_dir: Path | None = None,
    eddy_dir: Path | None = None,
    jhu_dir: Path | None = None,
) -> FSLContext:
    root = config.subject_output
    topup = topup_dir or root / "03_topup"
    bet = bet_dir or root / "04_bet"
    eddy = eddy_dir or root / "05_eddy"
    jhu = jhu_dir or root / "09_jhu_48roi"
    combined = len(audit.b0_indices) + audit.ap_b0_count
    eddy_prefix = eddy / "eddy_unwarped_images"
    return FSLContext(
        installation=installation,
        merged_b0=topup / "PA_AP_b0.nii.gz",
        merged_b0_shape=(*audit.pa_shape[:3], int(combined)),
        acqparams_topup=topup / "acqparams_topup.txt",
        topup_prefix=topup / "topup_PA_AP_b0",
        topup_corrected_b0s=topup / "topup_corrected_b0s",
        field_hz_prefix=topup / "topup_field_Hz",
        applytopup_inputs=(
            topup / "nodif_PA_all.nii.gz",
            topup / "nodif_AP_all.nii.gz",
        ),
        applytopup_indices=(1, len(audit.b0_indices) + 1),
        applytopup_output=topup / "applytopup_corrected_b0s",
        hifi_nodif=bet / "hifi_nodif",
        brain_prefix=bet / "hifi_nodif_brain",
        gibbs_pa=root / "02_gibbs" / "gibbs_PA.nii.gz",
        cleaned_mask=bet / "nodif_brain_mask.nii.gz",
        acqparams_eddy=topup / "acqparams_eddy.txt",
        index_eddy=topup / "index_eddy.txt",
        bvals=config.bvals,
        bvecs=root / "00_input_audit" / "bvecs_fsl_3xN",
        eddy_prefix=eddy_prefix,
        eddy_threads=min(12, max(1, os.cpu_count() or 1)),
        eddy_quad_output=eddy / "eddy_quad",
        subject_fa=root / "06_dti" / "FA.nii.gz",
        affine_fa=jhu / "FA_in_standard_affine.nii.gz",
        affine_matrix=jhu / "dti2standard_affine.mat",
        nonlinear_fa=jhu / "FA_in_standard_nonlinear.nii.gz",
        forward_warp=jhu / "dti2standard_warp.nii.gz",
        inverse_warp=jhu / "standard2dti_warp.nii.gz",
        atlas_labels=_ATLAS_IMAGE,
        subject_atlas=jhu / "WM_JHU_ROIs.nii.gz",
    )


def _model_context(config: PipelineConfig, work: Path) -> ModelContext:
    paths = _paths(config)
    return ModelContext(
        eddy_dwi=paths["eddy_dwi"],
        brain_mask=paths["brain_mask"],
        bvals=config.bvals,
        rotated_bvecs=paths["rotated_bvecs"],
        work_dir=work,
        henrique_helper=_HENRIQUE_HELPER,
        dti_max_b=config.analysis.dti_max_b,
    )


def _summary_metric_maps(paths: Mapping[str, Path]) -> Mapping[str, Path]:
    root = paths["dti_fa"].parents[1]
    values = {
        "DTI_FA": root / "06_dti" / "FA.nii.gz",
        "DTI_MD": root / "06_dti" / "MD.nii.gz",
        "DTI_AD": root / "06_dti" / "AD.nii.gz",
        "DTI_RD": root / "06_dti" / "RD.nii.gz",
        "DKI_FA": root / "07_dki" / "FA.nii.gz",
        "DKI_MD": root / "07_dki" / "MD.nii.gz",
        "DKI_AD": root / "07_dki" / "AD.nii.gz",
        "DKI_RD": root / "07_dki" / "RD.nii.gz",
        "DKI_MK": root / "07_dki" / "MK.nii.gz",
        "DKI_AK": root / "07_dki" / "AK.nii.gz",
        "DKI_RK": root / "07_dki" / "RK.nii.gz",
        "DKI_DIRECT_MD": root / "07_dki_direct" / "MD.nii.gz",
        "DKI_DIRECT_MK": root / "07_dki_direct" / "MK.nii.gz",
        "DKI_DIRECT_S0": root / "07_dki_direct" / "S0.nii.gz",
        "NODDI_ODI": root / "08_noddi" / "NODDI_odi.nii",
        "NODDI_FICVF": root / "08_noddi" / "NODDI_ficvf.nii",
        "NODDI_FISO": root / "08_noddi" / "NODDI_fiso.nii",
    }
    if tuple(values) != CANONICAL_METRICS:
        raise AssertionError("summary metric mapping changed")
    return MappingProxyType(values)


def _qc_context(
    config: PipelineConfig,
    work: Path,
    paths: Mapping[str, Path],
    software: Mapping[str, str],
) -> StageQCContext:
    root = config.subject_output
    return StageQCContext(
        stage_context=StageContext(
            config, _PACKAGE_ROOT, root, software
        ),
        output_directory=work,
        bvals=config.bvals,
        raw_pa=config.dwi_pa,
        raw_ap=config.b0_ap,
        stripe_metrics_csv=paths["stripe_csv"],
        stripe_decision_json=paths["stripe_decision"],
        denoised_pa=paths["denoised_pa"],
        denoised_ap=paths["denoised_ap"],
        gibbs_pa=paths["gibbs_pa"],
        gibbs_ap=paths["gibbs_ap"],
        topup_merged_b0=paths["merged_b0"],
        topup_corrected_b0=paths["topup_corrected"],
        topup_manifest_json=paths["topup_manifest"],
        hifi_b0=paths["hifi_b0"],
        brain_mask=paths["brain_mask"],
        eddy_dwi=paths["eddy_dwi"],
        eddy_parameters=paths["eddy_parameters"],
        eddy_movement_rms=paths["eddy_rms"],
        eddy_outlier_map=paths["eddy_outlier_map"],
        dti_maps={key: root / "06_dti" / f"{key}.nii.gz" for key in DTI_KEYS},
        dki_maps={key: root / "07_dki" / f"{key}.nii.gz" for key in DKI_KEYS},
        dki_direct_maps={
            key: root / "07_dki_direct" / f"{key}.nii.gz"
            for key in DKI_DIRECT_KEYS
        },
        noddi_maps={
            "ODI": root / "08_noddi" / "NODDI_odi.nii",
            "FICVF": root / "08_noddi" / "NODDI_ficvf.nii",
            "FISO": root / "08_noddi" / "NODDI_fiso.nii",
        },
        warped_atlas=paths["subject_atlas"],
    )


def _report_context(
    config: PipelineConfig,
    work: Path,
    paths: Mapping[str, Path],
    software: Mapping[str, str],
    *,
    stripe_detail_paths: Sequence[Path],
) -> ReportContext:
    root = config.subject_output
    records = tuple(root / name / ".stage_complete.json" for name in REPORT_STAGE_ORDER)
    return ReportContext(
        stage_context=StageContext(
            config, _PACKAGE_ROOT, root, software
        ),
        output_directory=work,
        qc_manifest_json=paths["qc_manifest"],
        input_audit_json=paths["audit_json"],
        stripe_metrics_csv=paths["stripe_csv"],
        stripe_decision_json=paths["stripe_decision"],
        stripe_detail_files=tuple(stripe_detail_paths),
        stage_metrics_json={
            "denoise": paths["denoise_metrics"],
            "gibbs": paths["gibbs_metrics"],
            "topup": paths["topup_metrics"],
            "bet": paths["bet_metrics"],
            "dti": paths["dti_metrics"],
            "dki": paths["dki_metrics"],
            "dki_direct": paths["direct_metrics"],
            "noddi": paths["noddi_metrics"],
        },
        topup_field_hz=paths["topup_field"],
        brain_mask=paths["brain_mask"],
        eddy_parameters=paths["eddy_parameters"],
        eddy_movement_rms=paths["eddy_rms"],
        eddy_outlier_map=paths["eddy_outlier_map"],
        eddy_outlier_report=paths["eddy_outlier_report"],
        eddy_quad_json=paths["eddy_quad_json"],
        noddi_error_codes=paths["noddi_error"],
        summary_json=paths["summary_json"],
        global_csv=paths["summary_global"],
        roi_csv=paths["summary_roi"],
        atlas_provenance_json=_ATLAS_PROVENANCE,
        stage_records=records,
    )


def _qc_input_paths(
    config: PipelineConfig, paths: Mapping[str, Path]
) -> tuple[Path, ...]:
    context_paths = [
        config.bvals,
        config.dwi_pa,
        config.b0_ap,
        paths["stripe_csv"],
        paths["stripe_decision"],
        paths["denoised_pa"],
        paths["denoised_ap"],
        paths["gibbs_pa"],
        paths["gibbs_ap"],
        paths["merged_b0"],
        paths["topup_corrected"],
        paths["topup_manifest"],
        paths["hifi_b0"],
        paths["brain_mask"],
        paths["eddy_dwi"],
        paths["eddy_parameters"],
        paths["eddy_rms"],
        paths["eddy_outlier_map"],
        paths["subject_atlas"],
        *_summary_metric_maps(paths).values(),
    ]
    return tuple(dict.fromkeys(context_paths))


def _report_input_paths(
    config: PipelineConfig,
    paths: Mapping[str, Path],
    *,
    stripe_detail_paths: Sequence[Path],
) -> tuple[Path, ...]:
    root = config.subject_output
    qc_figures = tuple(
        root
        / "qc"
        / filename.replace("${subject_id}", config.subject_id)
        for filename in FIGURE_FILENAMES.values()
    )
    fixed = [
        paths["qc_manifest"],
        *qc_figures,
        paths["audit_json"],
        paths["stripe_csv"],
        paths["stripe_decision"],
        *stripe_detail_paths,
        paths["denoise_metrics"],
        paths["gibbs_metrics"],
        paths["topup_metrics"],
        paths["bet_metrics"],
        paths["dti_metrics"],
        paths["dki_metrics"],
        paths["direct_metrics"],
        paths["noddi_metrics"],
        paths["topup_field"],
        paths["brain_mask"],
        paths["eddy_parameters"],
        paths["eddy_rms"],
        paths["eddy_outlier_map"],
        paths["eddy_outlier_report"],
        paths["eddy_quad_json"],
        paths["noddi_error"],
        paths["summary_json"],
        paths["summary_global"],
        paths["summary_roi"],
        _ATLAS_PROVENANCE,
        *(root / stage / ".stage_complete.json" for stage in REPORT_STAGE_ORDER),
    ]
    return tuple(dict.fromkeys(fixed))


def _write_canonical_bvecs(source: Path, destination: Path, volumes: int) -> None:
    values = _load_text(source, "b-vectors")
    try:
        canonical = normalize_bvecs(values, volumes)
    except Exception as error:
        raise ValueError(f"b-vectors must be finite and orientable as 3xN: {error}") from error
    if canonical.shape != (3, volumes) or not np.isfinite(canonical).all():
        raise ValueError("b-vectors must be finite and orientable as 3xN")
    np.savetxt(destination, canonical, fmt="%.10g")


def _validate_subject_atlas(atlas_path: Path, reference_fa: Path) -> None:
    try:
        atlas = nib.load(atlas_path)
        reference = nib.load(reference_fa)
        values = np.asarray(atlas.dataobj, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError("cannot read subject-space atlas or DTI FA") from error
    if (
        tuple(atlas.shape) != tuple(reference.shape)
        or not np.allclose(atlas.affine, reference.affine, atol=1e-5, rtol=0.0)
    ):
        raise ValueError("subject-space atlas must share the DTI FA grid and affine")
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("subject-space atlas must contain finite integral labels")
    nonzero = set(int(value) for value in np.unique(values) if value != 0)
    if nonzero != set(range(1, 49)):
        raise ValueError(
            "subject-space atlas nonzero labels must be exactly 1 through 48"
        )


def _parse_memtotal_kib(text: str) -> int:
    records = [line for line in text.splitlines() if line.startswith("MemTotal:")]
    if len(records) != 1:
        raise NODDIError("/proc/meminfo must contain exactly one MemTotal value in kB")
    match = re.fullmatch(r"MemTotal:[ \t]+([0-9]+)[ \t]+kB[ \t]*", records[0])
    if match is None:
        raise NODDIError("/proc/meminfo must contain exactly one MemTotal value in kB")
    value = int(match.group(1))
    if value <= 0:
        raise NODDIError("/proc/meminfo MemTotal must be positive")
    return value


def _installed_memory_gib(
    meminfo_path: Path = Path("/proc/meminfo"),
) -> float:
    try:
        text = meminfo_path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise NODDIError(f"cannot read physical memory from {meminfo_path}") from error
    kib = _parse_memtotal_kib(text)
    gib = kib / (1024.0**2)
    if not math.isfinite(gib) or gib <= 0:
        raise NODDIError("installed physical memory must be positive and finite")
    return gib


def _pipeline_noddi_workers(config: PipelineConfig) -> int:
    try:
        memory_gib = _installed_memory_gib()
    except NODDIError as error:
        raise PipelineExternalError(str(error)) from error
    try:
        return choose_noddi_workers(
            max(1, os.cpu_count() or 1),
            memory_gib,
            config.analysis.noddi_workers,
        )
    except NODDIError as error:
        raise PipelineInputError(str(error)) from error


def _write_topup_metrics(context: FSLContext, destination: Path) -> None:
    corrected = _load_nifti(context.topup_corrected_b0s.with_suffix(".nii.gz"))
    field = _load_nifti(context.field_hz_image)
    values = np.asarray(field.dataobj, dtype=np.float64)
    if len(corrected.shape) != 4 or len(field.shape) != 3 or not np.isfinite(values).all():
        raise StageStateError("TOPUP corrected images and field must be finite")
    payload = {
        "corrected_b0_count": int(corrected.shape[3]),
        "field_min_hz": float(np.min(values)),
        "field_max_hz": float(np.max(values)),
        "field_mean_hz": float(np.mean(values)),
    }
    _write_json(destination, payload)


def _write_eddy_quad_json(
    source_directory: Path,
    destination: Path,
    *,
    expected_volume_count: int | None = None,
    expected_slices: int | None = None,
    bvals: np.ndarray | None = None,
    outliers: np.ndarray | None = None,
    expected_paths: Mapping[str, Path] | None = None,
) -> None:
    if not source_directory.is_dir() or source_directory.is_symlink():
        raise StageStateError("EDDY QUAD output directory is missing or unsafe")
    source_files = _regular_tree_files(
        source_directory, require_single_link=True
    )
    selected = _eddy_quad_metrics(
        source_directory,
        source_files,
        expected_volume_count=expected_volume_count,
        expected_slices=expected_slices,
        bvals=bvals,
        outliers=outliers,
        expected_paths=expected_paths,
    )
    _write_json(
        destination,
        {
            "metrics": selected,
            "provenance": {
                "vols_no_outliers": _eddy_quad_no_outlier_source(
                    source_directory, source_files
                )
            },
        },
    )


def _validate_topup_outputs(
    work: Path,
    audit: InputAudit | None = None,
    config: PipelineConfig | None = None,
) -> Sequence[Path]:
    names = (
        "nodif_PA_all.nii.gz",
        "nodif_AP_all.nii.gz",
        "PA_AP_b0.nii.gz",
        "acqparams_topup.txt",
        "acqparams_eddy.txt",
        "index_eddy.txt",
        "bvals_rounded",
        "topup_input_manifest.json",
        "topup_PA_AP_b0_fieldcoef.nii.gz",
        "topup_PA_AP_b0_movpar.txt",
        "topup_corrected_b0s.nii.gz",
        "topup_field_Hz.nii.gz",
        "topup_metrics.json",
        "topup_fsl.log",
    )
    outputs = _required_outputs(work, names)
    if audit is None:
        raise StageStateError("TOPUP validation requires the fresh input audit")

    expected_affine = np.asarray(audit.pa_affine, dtype=np.float64)
    spatial = tuple(audit.pa_shape[:3])
    pa_count = len(audit.b0_indices)
    ap_count = audit.ap_b0_count
    combined = pa_count + ap_count
    pa = _validate_finite_nifti(
        work / "nodif_PA_all.nii.gz",
        "TOPUP PA b0 input",
        (*spatial, pa_count),
        expected_affine,
    )
    ap = _validate_finite_nifti(
        work / "nodif_AP_all.nii.gz",
        "TOPUP AP b0 input",
        (*spatial, ap_count),
        expected_affine,
    )
    merged = _validate_finite_nifti(
        work / "PA_AP_b0.nii.gz",
        "TOPUP merged b0 input",
        (*spatial, combined),
        expected_affine,
    )
    _validate_concatenated_nifti_volumes(
        pa,
        ap,
        merged,
        "TOPUP merged b0 volume order",
    )
    _validate_finite_nifti(
        work / "topup_corrected_b0s.nii.gz",
        "TOPUP corrected b0 output",
        (*spatial, combined),
        expected_affine,
    )

    fieldcoef = _validate_finite_nifti(
        work / "topup_PA_AP_b0_fieldcoef.nii.gz",
        "TOPUP field coefficients",
        None,
        None,
    )
    if len(fieldcoef.shape) not in (3, 4) or any(
        size < 1 for size in fieldcoef.shape
    ):
        raise StageStateError(
            "TOPUP field coefficients must be a nonempty 3D or 4D image"
        )
    field = _validate_finite_nifti(
        work / "topup_field_Hz.nii.gz",
        "TOPUP field image",
        spatial,
        expected_affine,
    )
    if len(field.shape) != 3:
        raise StageStateError("TOPUP field image must be three-dimensional")

    movpar = _finite_table(
        work / "topup_PA_AP_b0_movpar.txt",
        "TOPUP movement parameters",
        combined,
        6,
    )
    if movpar.shape != (combined, 6):
        raise StageStateError(
            "TOPUP movement parameters must contain one six-value row per b0"
        )
    acqparams = _finite_table(
        work / "acqparams_topup.txt",
        "TOPUP acquisition parameters",
        combined,
        4,
    )
    if acqparams.shape != (combined, 4):
        raise StageStateError(
            "TOPUP acquisition parameters must contain one four-value row per b0"
        )
    eddy_acqparams = _finite_table(
        work / "acqparams_eddy.txt",
        "EDDY acquisition parameters",
        2,
        4,
    )
    if eddy_acqparams.shape != (2, 4):
        raise StageStateError("EDDY acquisition parameters must contain two rows")
    if config is not None:
        pa_row = np.asarray(
            (
                *config.acquisition.pa_vector,
                config.acquisition.total_readout_time,
            ),
            dtype=np.float64,
        )
        ap_row = np.asarray(
            (
                *config.acquisition.ap_vector,
                config.acquisition.total_readout_time,
            ),
            dtype=np.float64,
        )
        expected_topup = np.vstack(
            (
                np.repeat(pa_row[None, :], pa_count, axis=0),
                np.repeat(ap_row[None, :], ap_count, axis=0),
            )
        )
        if not np.allclose(
            acqparams, expected_topup, atol=1e-10, rtol=0.0
        ) or not np.allclose(
            eddy_acqparams,
            np.vstack((pa_row, ap_row)),
            atol=1e-10,
            rtol=0.0,
        ):
            raise StageStateError(
                "TOPUP/EDDY acquisition rows must preserve exact PA/AP order"
            )
    index = _load_text(work / "index_eddy.txt", "EDDY index").reshape(-1)
    if (
        index.shape != (audit.pa_shape[3],)
        or not np.isfinite(index).all()
        or not np.equal(index, np.rint(index)).all()
        or not np.equal(index, 1).all()
    ):
        raise StageStateError(
            "EDDY index must contain one integral acquisition row per DWI volume"
        )
    rounded = _load_text(work / "bvals_rounded", "rounded b-values").reshape(-1)
    if rounded.shape != (audit.pa_shape[3],) or not np.isfinite(rounded).all():
        raise StageStateError(
            "rounded b-values must contain one finite value per DWI volume"
        )
    if config is not None:
        try:
            expected_rounded = round_shells(
                _load_text(config.bvals, "configured b-values").reshape(-1)
            )
        except InputAuditError as error:
            raise StageStateError("cannot validate rounded b-values") from error
        if not np.array_equal(rounded, expected_rounded):
            raise StageStateError(
                "rounded b-values do not match the configured DWI protocol"
            )
    manifest = _read_json(work / "topup_input_manifest.json", "TOPUP manifest")
    if (
        manifest.get("pa_b0_count") != pa_count
        or manifest.get("ap_b0_count") != ap_count
        or manifest.get("combined_b0_count") != combined
        or manifest.get("pa_b0_indices") != list(audit.b0_indices)
        or manifest.get("volume_order")
        != ["PA"] * pa_count + ["AP"] * ap_count
        or manifest.get("eddy_index_count") != audit.pa_shape[3]
    ):
        raise StageStateError(
            "TOPUP manifest does not describe the expected PA/AP volume order"
        )
    if config is not None and (
        manifest.get("pa_acquisition_row") != pa_row.tolist()
        or manifest.get("ap_acquisition_row") != ap_row.tolist()
        or manifest.get("eddy_acquisition_row_order") != ["PA", "AP"]
    ):
        raise StageStateError(
            "TOPUP manifest acquisition metadata does not preserve PA/AP order"
        )
    metrics = _read_json(work / "topup_metrics.json", "TOPUP metrics")
    if metrics.get("corrected_b0_count") != combined:
        raise StageStateError("TOPUP metrics corrected-b0 count is inconsistent")
    return outputs


def _validate_bet_outputs(work: Path) -> Sequence[Path]:
    return _required_outputs(
        work,
        (
            "hifi_nodif.nii.gz",
            "hifi_nodif_brain.nii.gz",
            "hifi_nodif_brain_mask.nii.gz",
            "nodif_brain_mask.nii.gz",
            "mask_cleanup_metrics.json",
            "bet_fsl.log",
        ),
    )


def _validate_eddy_outputs(
    work: Path,
    audit: InputAudit | None = None,
    bvals_path: Path | None = None,
    *,
    slice_axis: int = 2,
    expected_mask_path: Path | None = None,
) -> Sequence[Path]:
    prefix = work / "eddy_unwarped_images"
    required = (
        Path(f"{prefix}.nii.gz"),
        Path(f"{prefix}.eddy_rotated_bvecs"),
        Path(f"{prefix}.eddy_parameters"),
        Path(f"{prefix}.eddy_movement_rms"),
        Path(f"{prefix}.eddy_restricted_movement_rms"),
        Path(f"{prefix}.eddy_outlier_map"),
        Path(f"{prefix}.eddy_outlier_report"),
        Path(f"{prefix}.eddy_residuals.nii.gz"),
        Path(f"{prefix}.eddy_cnr_maps.nii.gz"),
        work / "eddy_quad.json",
        work / "eddy_fsl.log",
    )
    _require_regular_files(required)
    if audit is None or bvals_path is None:
        raise StageStateError("EDDY validation requires the fresh input audit and b-values")
    expected_shape = tuple(audit.pa_shape)
    expected_affine = np.asarray(audit.pa_affine, dtype=np.float64)
    volumes = expected_shape[3]
    _validate_finite_nifti(
        required[0], "EDDY corrected DWI", expected_shape, expected_affine
    )

    bvals = _load_text(bvals_path, "b-values").reshape(-1)
    if bvals.shape != (volumes,) or not np.isfinite(bvals).all():
        raise StageStateError("EDDY b-values must contain one finite value per volume")
    rotated = _load_text(required[1], "EDDY rotated b-vectors")
    if rotated.shape != (3, volumes) or not np.isfinite(rotated).all():
        raise StageStateError("EDDY rotated b-vectors must be finite exact 3xN")
    norms = np.linalg.norm(rotated, axis=0)
    b0 = bvals < 50.0
    accepted_b0 = (norms[b0] < 0.1) | (
        (norms[b0] >= 0.95) & (norms[b0] <= 1.05)
    )
    if not accepted_b0.all():
        raise StageStateError("EDDY rotated b0 vectors must be near-zero or unit norm")
    if np.any((norms[~b0] < 0.95) | (norms[~b0] > 1.05)):
        raise StageStateError("EDDY rotated non-b0 vectors must have unit norm")

    parameters = _finite_table(
        required[2], "EDDY parameters", volumes, 6
    )
    movement = _finite_table(
        required[3], "EDDY movement RMS", volumes, 2
    )
    restricted = _finite_table(
        required[4], "EDDY restricted movement RMS", volumes, 2
    )
    if parameters.shape[1] < 6:
        raise StageStateError(
            "EDDY parameters must have N rows and at least six finite columns"
        )
    if movement.shape[1] < 2 or restricted.shape[1] < 2:
        raise StageStateError(
            "EDDY RMS files must have N rows and at least two finite columns"
        )
    if np.any(movement < 0) or np.any(restricted < 0):
        raise StageStateError("EDDY RMS values must be nonnegative")

    outliers = _load_eddy_outlier_map(required[5], volumes)
    expected_slices = expected_shape[slice_axis]
    if outliers.shape[1] != expected_slices:
        raise StageStateError(
            "EDDY outlier map column count must equal the acquisition slice count"
        )
    reported = _eddy_reported_outlier_count(required[6])
    observed = int(np.sum(outliers))
    if reported != observed:
        raise StageStateError(
            "EDDY outlier report count is inconsistent with the binary outlier map"
        )

    _validate_finite_nifti(
        required[7], "EDDY residual image", expected_shape, expected_affine
    )
    cnr = _validate_finite_nifti(
        required[8], "EDDY CNR maps", None, expected_affine
    )
    if (
        len(cnr.shape) != 4
        or tuple(cnr.shape[:3]) != expected_shape[:3]
        or cnr.shape[3] < 1
    ):
        raise StageStateError(
            "EDDY CNR maps must be finite 4D data on the expected spatial grid"
        )

    quad_root = work / "eddy_quad"
    quad_files = _regular_tree_files(quad_root, require_single_link=True)
    expected_quad_paths = {
        "data_file_eddy": required[0],
        "data_file_bvals": bvals_path,
        "qc_path": quad_root,
    }
    if expected_mask_path is not None:
        expected_quad_paths["data_file_mask"] = expected_mask_path
    expected_metrics = _eddy_quad_metrics(
        quad_root,
        quad_files,
        expected_volume_count=volumes,
        expected_slices=expected_slices,
        bvals=bvals,
        outliers=outliers,
        expected_paths=expected_quad_paths,
    )
    sanitized = _read_json(work / "eddy_quad.json", "sanitized EDDY QUAD JSON")
    expected_provenance = {
        "vols_no_outliers": _eddy_quad_no_outlier_source(
            quad_root, quad_files
        )
    }
    if (
        sanitized
        != {
            "metrics": expected_metrics,
            "provenance": expected_provenance,
        }
    ):
        raise StageStateError(
            "sanitized EDDY QUAD metrics/provenance do not match "
            "the validated source tree"
        )
    return (*required, *quad_files)


def _validate_noddi_outputs(work: Path) -> Sequence[Path]:
    required = (
        "NODDI_odi.nii",
        "NODDI_ficvf.nii",
        "NODDI_fiso.nii",
        "NODDI_kappa.nii",
        "NODDI_fmin.nii",
        "NODDI_error_code.nii",
        "NODDI_fibredirs_xvec.nii",
        "NODDI_fibredirs_yvec.nii",
        "NODDI_fibredirs_zvec.nii",
        "NODDI_params.mat",
        "noddi_metrics.json",
        "noddi_prepare.json",
        "noddi_prepare.log",
        "merge_noddi.log",
    )
    _required_outputs(work, required)
    return _regular_tree_files(work)


def _validate_jhu_outputs(work: Path) -> Sequence[Path]:
    required = _required_outputs(
        work,
        (
            "FA_in_standard_affine.nii.gz",
            "dti2standard_affine.mat",
            "FA_in_standard_nonlinear.nii.gz",
            "dti2standard_warp.nii.gz",
            "standard2dti_warp.nii.gz",
            "WM_JHU_ROIs.nii.gz",
            "jhu_registration_fsl.log",
        ),
    )
    _validate_subject_atlas(
        work / "WM_JHU_ROIs.nii.gz",
        work.parents[1] / "06_dti" / "FA.nii.gz",
    )
    return required


def _validate_finite_nifti(
    path: Path,
    label: str,
    expected_shape: tuple[int, ...] | None,
    expected_affine: np.ndarray | None,
) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(path)
        affine = np.asarray(image.affine, dtype=np.float64)
        shape = tuple(int(size) for size in image.shape)
    except Exception as error:
        raise StageStateError(f"cannot read {label}: {path.name}") from error
    if not np.isfinite(affine).all():
        raise StageStateError(f"{label} must contain a finite affine")
    if expected_shape is not None and shape != tuple(expected_shape):
        raise StageStateError(
            f"{label} has the wrong shape: expected {expected_shape}, "
            f"observed {shape}"
        )
    if expected_affine is not None and not np.allclose(
        affine, np.asarray(expected_affine), atol=1e-5, rtol=0.0
    ):
        raise StageStateError(f"{label} must use the expected image grid/affine")
    if not shape or any(size < 1 for size in shape):
        raise StageStateError(f"{label} must be a nonempty image")
    if len(shape) not in (3, 4):
        raise StageStateError(f"{label} must be a three- or four-dimensional image")
    try:
        if len(shape) == 3:
            chunks = (np.asarray(image.dataobj, dtype=np.float64),)
        else:
            chunks = (
                np.asarray(image.dataobj[..., volume], dtype=np.float64)
                for volume in range(shape[3])
            )
        for chunk in chunks:
            if tuple(chunk.shape) != shape[:3] or not np.isfinite(chunk).all():
                raise StageStateError(f"{label} must contain finite image data")
    except StageStateError:
        raise
    except Exception as error:
        raise StageStateError(f"cannot read {label}: {path.name}") from error
    return image


def _validate_concatenated_nifti_volumes(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    merged: nib.spatialimages.SpatialImage,
    label: str,
) -> None:
    merged_volume = 0
    try:
        for source in (first, second):
            for source_volume in range(source.shape[3]):
                expected = np.asarray(
                    source.dataobj[..., source_volume], dtype=np.float64
                )
                observed = np.asarray(
                    merged.dataobj[..., merged_volume], dtype=np.float64
                )
                if expected.shape != observed.shape or not np.array_equal(
                    expected, observed
                ):
                    raise StageStateError(
                        f"{label} must be the exact PA/AP concatenation"
                    )
                merged_volume += 1
    except StageStateError:
        raise
    except Exception as error:
        raise StageStateError(f"cannot validate {label}") from error


def _finite_table(
    path: Path,
    label: str,
    expected_rows: int,
    minimum_columns: int,
) -> np.ndarray:
    values = _load_text(path, label)
    if values.ndim == 1:
        values = values[None, :]
    if (
        values.ndim != 2
        or values.shape[0] != expected_rows
        or values.shape[1] < minimum_columns
        or not np.isfinite(values).all()
    ):
        raise StageStateError(
            f"{label} must contain {expected_rows} finite rows and at least "
            f"{minimum_columns} columns"
        )
    return values


def _load_eddy_outlier_map(path: Path, volumes: int) -> np.ndarray:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as error:
        raise StageStateError("cannot read EDDY outlier map") from error
    if not lines:
        raise StageStateError("EDDY outlier map is empty")
    tokens = lines[0].split()
    try:
        [float(token) for token in tokens]
        data_lines = lines
    except ValueError:
        data_lines = lines[1:]
    try:
        rows = [[float(token) for token in line.split()] for line in data_lines]
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("ragged or empty map")
        values = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StageStateError("EDDY outlier map contains malformed values") from error
    if values.ndim != 2 or values.shape[0] != volumes or values.shape[1] < 1:
        raise StageStateError(
            "EDDY outlier map must have one row per DWI volume"
        )
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.rint(values)).all()
        or not np.isin(values, (0, 1)).all()
    ):
        raise StageStateError(
            "EDDY outlier map must contain finite integral binary 0/1 values"
        )
    return values.astype(np.uint8)


def _eddy_reported_outlier_count(path: Path) -> int:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as error:
        raise StageStateError("cannot read EDDY outlier report") from error
    pattern = re.compile(
        r"Slice\s+\d+\s+in\s+scan\s+\d+\s+is\s+an\s+outlier\b.*",
        flags=re.IGNORECASE,
    )
    if not all(pattern.fullmatch(line) for line in lines):
        raise StageStateError("unsupported EDDY outlier report syntax")
    return len(lines)


def _eddy_quad_no_outlier_source(
    root: Path, files: Sequence[Path]
) -> str:
    source = (
        "eddy-quad-legacy-residual-msr"
        if root / "vols_no_outliers.txt" in set(files)
        else "not-emitted-by-eddy-quad"
    )
    if source not in _EDDY_QUAD_NO_OUTLIER_SOURCES:
        raise StageStateError("EDDY QUAD no-outlier provenance is invalid")
    return source


def _eddy_quad_shell_tolerance(payload: Mapping[str, object]) -> float:
    tolerance = 100.0
    eddy_input = payload["eddy_input"]
    if isinstance(eddy_input, dict):
        try:
            tolerance = float(eddy_input["b_range"])
        except (KeyError, TypeError, ValueError):
            tolerance = 100.0
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise StageStateError("EDDY QUAD b_range is invalid")
    return tolerance


def _eddy_quad_rounded_bvals(
    bvals: np.ndarray, tolerance: float
) -> np.ndarray:
    """Mirror eddy_qc's transitive grouping and integer-median shell labels."""
    selected = np.zeros(bvals.size, dtype=bool)
    same_shell = np.abs(bvals[:, None] - bvals[None, :]) <= tolerance
    rounded = bvals.copy()
    while not selected.all():
        use = np.zeros(selected.size, dtype=bool)
        use[np.flatnonzero(~selected)[0]] = True
        previous_count = 0
        while int(np.count_nonzero(use)) != previous_count:
            previous_count = int(np.count_nonzero(use))
            use = same_shell[use].any(axis=0)
            if np.any(use & selected):
                raise StageStateError(
                    "EDDY QUAD shell grouping reused a selected volume"
                )
        rounded[use] = int(np.median(bvals[use]))
        selected[use] = True
    rounded[rounded <= tolerance] = 0
    return rounded


def _eddy_quad_metrics(
    root: Path,
    files: Sequence[Path] | None = None,
    *,
    expected_volume_count: int | None = None,
    expected_slices: int | None = None,
    bvals: np.ndarray | None = None,
    outliers: np.ndarray | None = None,
    expected_paths: Mapping[str, Path] | None = None,
) -> dict[str, float]:
    candidates = _regular_tree_files(root, require_single_link=True)
    if files is not None and tuple(files) != candidates:
        raise StageStateError("EDDY QUAD source tree identity changed during validation")
    required = {
        name: root / name
        for name in ("qc.json", "qc.pdf", "eddy_msr.txt")
    }
    candidate_set = set(candidates)
    for name, path in required.items():
        if path not in candidate_set:
            raise StageStateError(f"EDDY QUAD required product is missing: {name}")
    _validate_eddy_quad_pdf(required["qc.pdf"])
    payload = _read_unique_json(required["qc.json"], "EDDY QUAD qc.json")
    if set(payload) != _EDDY_QUAD_SCHEMA:
        missing = sorted(_EDDY_QUAD_SCHEMA - set(payload))
        unexpected = sorted(set(payload) - _EDDY_QUAD_SCHEMA)
        raise StageStateError(
            "EDDY QUAD qc.json schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    _validate_eddy_input_payload(payload)
    paths = dict(expected_paths or {})
    paths.setdefault("qc_path", root)
    for key in ("data_file_eddy", "data_file_mask", "data_file_bvals", "qc_path"):
        _validate_eddy_quad_path(payload[key], key, paths.get(key))

    no_dw = _eddy_quad_integer(payload["data_no_dw_vols"], "data_no_dw_vols", 1)
    no_b0 = _eddy_quad_integer(payload["data_no_b0_vols"], "data_no_b0_vols", 1)
    no_pe = _eddy_quad_integer(payload["data_no_PE_dirs"], "data_no_PE_dirs", 1)
    no_shells = _eddy_quad_integer(payload["data_no_shells"], "data_no_shells", 1)
    volume_count = no_dw + no_b0
    if expected_volume_count is not None and volume_count != expected_volume_count:
        raise StageStateError("EDDY QUAD volume count is inconsistent")
    if no_pe != 1:
        raise StageStateError("EDDY QUAD pipeline contract requires one indexed PE group")
    _read_eddy_quad_msr(required["eddy_msr.txt"], volume_count)

    protocol = _eddy_quad_vector(
        payload["data_protocol"],
        "data_protocol",
        no_shells + 1,
        integral=True,
        minimum=0.0,
    )
    if int(np.sum(protocol)) != volume_count:
        raise StageStateError("EDDY QUAD protocol volume count is inconsistent")
    unique_bvals = _eddy_quad_vector(
        payload["data_unique_bvals"],
        "data_unique_bvals",
        no_shells,
        minimum=np.finfo(float).tiny,
    )
    if not np.all(np.diff(unique_bvals) > 0):
        raise StageStateError("EDDY QUAD unique b-values must be strictly increasing")
    unique_pes = _eddy_quad_vector(
        payload["data_unique_pes"],
        "data_unique_pes",
        no_pe,
        integral=True,
        minimum=1.0,
    )
    if not np.array_equal(unique_pes, np.arange(1, no_pe + 1)):
        raise StageStateError("EDDY QUAD PE indices must be consecutive and one-based")
    eddy_parameters = _eddy_quad_vector(
        payload["data_eddy_para"],
        "data_eddy_para",
        no_pe * 4,
    )
    if np.any(eddy_parameters[3::4] <= 0):
        raise StageStateError("EDDY QUAD readout times must be positive")
    _eddy_quad_vector(
        payload["data_vox_size"],
        "data_vox_size",
        3,
        minimum=np.finfo(float).tiny,
    )

    if bvals is not None:
        expected_bvals = np.asarray(bvals, dtype=np.float64).reshape(-1)
        if (
            expected_bvals.shape != (volume_count,)
            or not np.isfinite(expected_bvals).all()
        ):
            raise StageStateError("EDDY QUAD b-values are inconsistent")
        shell_tolerance = _eddy_quad_shell_tolerance(payload)
        diffusion = expected_bvals > shell_tolerance
        if int(np.count_nonzero(diffusion)) != no_dw:
            raise StageStateError("EDDY QUAD diffusion volume count is inconsistent")
        expected_shells = _eddy_quad_rounded_bvals(
            expected_bvals, shell_tolerance
        )
        all_shell_values = np.unique(expected_shells.astype(int))
        shell_values = all_shell_values[
            all_shell_values > shell_tolerance
        ]
        if not np.array_equal(unique_bvals, shell_values.astype(float)):
            raise StageStateError("EDDY QUAD shell values are inconsistent")
        expected_protocol = np.asarray(
            [
                int(np.count_nonzero(expected_shells == shell))
                for shell in all_shell_values
            ],
            dtype=np.float64,
        )
        if not np.array_equal(protocol, expected_protocol):
            raise StageStateError("EDDY QUAD protocol counts are inconsistent")
    else:
        expected_bvals = None
        expected_shells = None
        shell_values = unique_bvals

    flags = (
        "qc_params_flag",
        "qc_s2v_params_flag",
        "qc_field_flag",
        "qc_ol_flag",
        "qc_cnr_flag",
        "qc_rss_flag",
    )
    if any(type(payload[key]) is not bool for key in flags):
        raise StageStateError("EDDY QUAD QC flags must be JSON booleans")
    if not payload["qc_params_flag"]:
        raise StageStateError("EDDY QUAD parameter metrics are required")
    if payload["qc_s2v_params_flag"]:
        raise StageStateError("EDDY QUAD unexpected slice-to-volume metrics")
    if not all(
        bool(payload[key])
        for key in ("qc_field_flag", "qc_ol_flag", "qc_cnr_flag", "qc_rss_flag")
    ):
        raise StageStateError(
            "EDDY QUAD field/outlier/CNR/residual metrics are required"
        )

    motion_abs = _eddy_quad_number(payload["qc_mot_abs"], "qc_mot_abs", 0.0)
    motion_rel = _eddy_quad_number(payload["qc_mot_rel"], "qc_mot_rel", 0.0)
    params = _eddy_quad_vector(payload["qc_params_avg"], "qc_params_avg", 9)
    s2v = _eddy_quad_vector(
        payload["qc_s2v_params_avg_std"],
        "qc_s2v_params_avg_std",
        6,
    )
    if not np.equal(s2v, -1.0).all():
        raise StageStateError(
            "EDDY QUAD disabled slice-to-volume metrics must use -1 placeholders"
        )
    displacement = _eddy_quad_number(
        payload["qc_vox_displ_std"],
        "qc_vox_displ_std",
        0.0,
    )
    outlier_total = _eddy_quad_number(
        payload["qc_outliers_tot"], "qc_outliers_tot", 0.0, 100.0
    )
    outlier_shells = _eddy_quad_vector(
        payload["qc_outliers_b"],
        "qc_outliers_b",
        no_shells,
        minimum=0.0,
        maximum=100.0,
    )
    outlier_pes = _eddy_quad_vector(
        payload["qc_outliers_pe"],
        "qc_outliers_pe",
        no_pe,
        minimum=0.0,
        maximum=100.0,
    )
    cnr_avg = _eddy_quad_vector(
        payload["qc_cnr_avg"],
        "qc_cnr_avg",
        no_shells + 1,
        minimum=0.0,
    )
    cnr_std = _eddy_quad_vector(
        payload["qc_cnr_std"],
        "qc_cnr_std",
        no_shells + 1,
        minimum=0.0,
    )

    legacy_volumes = root / "vols_no_outliers.txt"
    if legacy_volumes in candidate_set:
        _read_eddy_quad_no_outlier_volumes(legacy_volumes, volume_count)
    if outliers is not None:
        outlier_map = np.asarray(outliers)
        if (
            outlier_map.ndim != 2
            or outlier_map.shape[0] != volume_count
            or (
                expected_slices is not None
                and outlier_map.shape[1] != expected_slices
            )
            or not np.isin(outlier_map, (0, 1)).all()
        ):
            raise StageStateError("EDDY QUAD outlier map is inconsistent")
        slice_count = int(outlier_map.shape[1])
        observed = int(np.count_nonzero(outlier_map))
        expected_total = 100.0 * observed / (no_dw * slice_count)
        if not math.isclose(outlier_total, expected_total, abs_tol=1e-8):
            raise StageStateError("EDDY QUAD total outlier percentage is inconsistent")
        if expected_bvals is not None and expected_shells is not None:
            expected_by_shell = np.asarray(
                [
                    100.0
                    * np.count_nonzero(outlier_map[expected_shells == shell])
                    / (
                        np.count_nonzero(expected_shells == shell)
                        * slice_count
                    )
                    for shell in shell_values
                ]
            )
            if not np.allclose(
                outlier_shells, expected_by_shell, atol=1e-8, rtol=0.0
            ):
                raise StageStateError(
                    "EDDY QUAD shell outlier percentages are inconsistent"
                )
        expected_by_pe = np.asarray(
            [100.0 * observed / (volume_count * slice_count)]
        )
        if not np.allclose(outlier_pes, expected_by_pe, atol=1e-8, rtol=0.0):
            raise StageStateError(
                "EDDY QUAD PE outlier percentages are inconsistent"
            )
    selected = {
        "qc_mot_abs": motion_abs,
        "qc_mot_rel": motion_rel,
        "qc_outliers_tot": outlier_total,
        "qc_vox_displ_std": displacement,
    }
    for key, values in (
        ("qc_params_avg", params),
        ("qc_outliers_b", outlier_shells),
        ("qc_outliers_pe", outlier_pes),
        ("qc_cnr_avg", cnr_avg),
        ("qc_cnr_std", cnr_std),
    ):
        for index, value in enumerate(values):
            selected[f"{key}[{index}]"] = float(value)
    return dict(sorted(selected.items()))


def _read_unique_json(path: Path, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise StageStateError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except StageStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageStateError(f"cannot read {label}") from error
    if not isinstance(payload, dict):
        raise StageStateError(f"{label} must be a JSON object")
    _finite_json(payload, label)
    return payload


def _validate_eddy_input_payload(payload: Mapping[str, object]) -> None:
    flag = payload["eddy_input_flag"]
    if type(flag) is not bool:
        raise StageStateError("EDDY QUAD eddy_input_flag must be a JSON boolean")
    value = payload["eddy_input"]
    if not flag:
        if value is not False:
            raise StageStateError(
                "EDDY QUAD disabled eddy_input must be JSON false"
            )
        return
    if not isinstance(value, dict) or not value or len(value) > 256:
        raise StageStateError("EDDY QUAD eddy_input must be a bounded object")
    for key, child in value.items():
        if not isinstance(key, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,127}", key
        ):
            raise StageStateError("EDDY QUAD eddy_input contains an unsafe key")
        if not isinstance(child, str):
            raise StageStateError("EDDY QUAD eddy_input values must be strings")
        _validate_bounded_string(
            child,
            "EDDY QUAD eddy_input value",
            allow_empty=True,
        )


def _validate_bounded_string(
    value: str, label: str, *, allow_empty: bool = False
) -> None:
    if (
        (not value and not allow_empty)
        or len(value) > 8192
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise StageStateError(f"{label} is unsafe")


def _validate_eddy_quad_path(
    value: object, label: str, expected: Path | None
) -> None:
    if not isinstance(value, str):
        raise StageStateError(f"EDDY QUAD {label} must be a path string")
    _validate_bounded_string(value, f"EDDY QUAD {label}")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise StageStateError(f"EDDY QUAD {label} path traversal is unsafe")
    if expected is not None and Path(os.path.abspath(path)) != Path(
        os.path.abspath(expected)
    ):
        raise StageStateError(f"EDDY QUAD {label} path is inconsistent")


def _eddy_quad_integer(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StageStateError(f"EDDY QUAD {label} must be an integer >= {minimum}")
    return value


def _eddy_quad_number(
    value: object,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageStateError(f"EDDY QUAD {label} must be numeric")
    number = float(value)
    if (
        not math.isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise StageStateError(f"EDDY QUAD {label} is outside its valid range")
    return number


def _eddy_quad_vector(
    value: object,
    label: str,
    length: int,
    *,
    integral: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise StageStateError(
            f"EDDY QUAD {label} must contain exactly {length} values"
        )
    numbers = np.asarray(
        [
            _eddy_quad_number(child, f"{label}[{index}]", minimum, maximum)
            for index, child in enumerate(value)
        ],
        dtype=np.float64,
    )
    if integral and not np.equal(numbers, np.rint(numbers)).all():
        raise StageStateError(f"EDDY QUAD {label} values must be integral")
    return numbers


def _validate_eddy_quad_pdf(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_EDDY_QUAD_PDF_BYTES
        ):
            raise StageStateError(
                "EDDY QUAD qc.pdf must be a nonempty single-link regular PDF"
            )
        data = path.read_bytes()
    except StageStateError:
        raise
    except OSError as error:
        raise StageStateError("cannot read EDDY QUAD qc.pdf") from error
    if not re.match(rb"%PDF-1\.[0-9]\r?\n", data):
        raise StageStateError("EDDY QUAD qc.pdf has an invalid PDF header")
    match = re.search(rb"startxref\s+([0-9]+)\s+%%EOF\s*$", data)
    if match is None:
        raise StageStateError("EDDY QUAD qc.pdf is truncated or malformed")
    xref = int(match.group(1))
    if xref >= len(data) or data[xref : xref + 4] != b"xref":
        raise StageStateError("EDDY QUAD qc.pdf has an invalid startxref")
    for marker in (
        rb"/Type\s*/Catalog\b",
        rb"/Type\s*/Pages\b",
        rb"/Type\s*/Page\b",
    ):
        if re.search(marker, data) is None:
            raise StageStateError("EDDY QUAD qc.pdf lacks a required page structure")


def _read_bounded_eddy_quad_ascii(
    path: Path, label: str, maximum_bytes: int
) -> str:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise StageStateError(f"cannot inspect EDDY QUAD {label}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > min(maximum_bytes, _MAX_EDDY_QUAD_TEXT_BYTES)
    ):
        raise StageStateError(
            f"EDDY QUAD {label} exceeds its bounded text contract"
        )
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise StageStateError(f"cannot read EDDY QUAD {label}") from error
    if len(text.encode("ascii")) != metadata.st_size:
        raise StageStateError(f"EDDY QUAD {label} changed while reading")
    return text


def _read_eddy_quad_msr(path: Path, volume_count: int) -> np.ndarray:
    maximum_bytes = volume_count * 320 + max(0, volume_count - 1) + 1
    text = _read_bounded_eddy_quad_ascii(
        path, "eddy_msr.txt", maximum_bytes
    )
    number = r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}"
    if re.fullmatch(rf"{number}(?: {number})*\n", text) is None:
        raise StageStateError(
            "EDDY QUAD eddy_msr.txt is not a canonical producer row"
        )
    tokens = text[:-1].split(" ")
    if len(tokens) != volume_count:
        raise StageStateError(
            "EDDY QUAD eddy_msr.txt must contain one value per volume"
        )
    values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise StageStateError(
            "EDDY QUAD eddy_msr.txt values must be finite and nonnegative"
        )
    return values


def _read_eddy_quad_no_outlier_volumes(
    path: Path, volume_count: int
) -> tuple[int, ...]:
    maximum_digits = len(str(volume_count - 1))
    maximum_bytes = (
        volume_count * maximum_digits + max(0, volume_count - 1) + 1
    )
    text = _read_bounded_eddy_quad_ascii(
        path, "vols_no_outliers.txt", maximum_bytes
    )
    if re.fullmatch(
        r"(?:0|[1-9][0-9]*)(?: (?:0|[1-9][0-9]*))*\n",
        text,
    ) is None:
        raise StageStateError("EDDY QUAD vols_no_outliers.txt is malformed")
    tokens = text[:-1].split(" ")
    if (
        len(tokens) > volume_count
        or any(len(token) > maximum_digits for token in tokens)
    ):
        raise StageStateError(
            "EDDY QUAD vols_no_outliers.txt exceeds its volume bounds"
        )
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise StageStateError(
            "EDDY QUAD vols_no_outliers.txt is malformed"
        ) from error
    if (
        values != tuple(sorted(set(values)))
        or values[-1] >= volume_count
    ):
        raise StageStateError(
            "EDDY QUAD vols_no_outliers.txt must contain unique sorted volumes"
        )
    return values


def _summary_validator(config: PipelineConfig) -> Callable[[Path], Sequence[Path]]:
    def validate(work: Path) -> Sequence[Path]:
        paths = _required_outputs(
            work,
            (
                f"{config.subject_id}_JHU_48ROI_metrics.csv",
                f"{config.subject_id}_global_metrics.csv",
                f"{config.subject_id}_summary.json",
            ),
        )
        with paths[0].open(newline="", encoding="utf-8") as handle:
            roi_rows = list(csv.DictReader(handle))
        with paths[1].open(newline="", encoding="utf-8") as handle:
            global_rows = list(csv.DictReader(handle))
        if len(roi_rows) != 48 or [int(row["label_id"]) for row in roi_rows] != list(range(1, 49)):
            raise StageStateError("summary ROI CSV must contain true labels 1 through 48")
        if len(global_rows) != 1:
            raise StageStateError("summary global CSV must contain exactly one row")
        _read_json(paths[2], "summary JSON")
        return paths

    return validate


def _qc_validator(config: PipelineConfig) -> Callable[[Path], Sequence[Path]]:
    def validate(work: Path) -> Sequence[Path]:
        names = [
            (
                filename.replace("${subject_id}", config.subject_id)
            )
            for filename in FIGURE_FILENAMES.values()
        ]
        return _required_outputs(work, (*names, "qc_manifest.json"))

    return validate


def _report_validator(config: PipelineConfig) -> Callable[[Path], Sequence[Path]]:
    return lambda work: _required_outputs(
        work,
        (
            f"{config.subject_id}_QC_report.pdf",
            f"{config.subject_id}_analysis_report.md",
            f"{config.subject_id}_run_summary.json",
        ),
    )


def _dry_run(
    config: PipelineConfig,
    audit: InputAudit,
    plan: Sequence[StageSpec],
    runtime: _Runtime,
) -> tuple[PipelineStageOutcome, ...]:
    print(f"subject={config.subject_id}")
    for label, path in (
        ("dwi_pa", config.dwi_pa),
        ("bvals", config.bvals),
        ("bvecs", config.bvecs),
        ("b0_ap", config.b0_ap),
        ("subject_output", config.subject_output),
    ):
        print(f"{label}={path}")
    full_software = _software_provenance(runtime)
    print("software=" + json.dumps(dict(full_software), sort_keys=True))
    gate_runner = StageRunner(
        StageContext(
            config,
            _PACKAGE_ROOT,
            config.subject_output,
            _base_software_provenance(),
        )
    )
    scientific_runner = StageRunner(
        StageContext(
            config, _PACKAGE_ROOT, config.subject_output, full_software
        )
    )
    outcomes: list[PipelineStageOutcome] = []
    upstream_runnable = True
    for index, spec in enumerate(plan):
        runner = gate_runner if index < 2 else scientific_runner
        lexical_final = config.subject_output / spec.name
        lexical_work = config.subject_output / ".work" / spec.name
        if not upstream_runnable:
            status = "blocked"
        elif lexical_final.is_symlink():
            status = "stale"
            upstream_runnable = False
        elif runner.is_current(spec):
            status = "current/skipped"
        elif lexical_final.exists():
            status = "stale"
            upstream_runnable = False
        elif lexical_work.is_symlink() or (
            lexical_work.exists() and not lexical_work.is_dir()
        ):
            status = "blocked"
            upstream_runnable = False
        elif lexical_work.is_dir():
            try:
                nonempty_work = any(lexical_work.iterdir())
            except OSError:
                status = "blocked"
                upstream_runnable = False
            else:
                if nonempty_work and spec.name == "08_noddi":
                    status = "runnable/resumable"
                elif nonempty_work:
                    status = "blocked"
                    upstream_runnable = False
                else:
                    status = "runnable"
        elif upstream_runnable:
            status = "runnable"
        else:
            status = "blocked"
        print(f"{spec.name}: {status}")
        outcomes.append(
            PipelineStageOutcome(
                spec.name,
                status,
                config.subject_output / spec.name,
                config.subject_output / spec.name / ".stage_complete.json",
            )
        )
        if (
            spec.name == "00_pre_denoise_motion_qc"
            and status == "current/skipped"
        ):
            decision = _read_qc_decision(runner.final_dir(spec.name))
            print(f"{spec.name}: decision={decision.status}")
            if decision.status in {"EXCLUDE", "HOLD_FOR_REVIEW"}:
                upstream_runnable = False
            elif decision.status not in _CONTINUE_QC:
                raise StageStateError(
                    f"unsupported stripe-QC decision: {decision.status}"
                )
    for argv in _dry_run_commands(config, audit, runtime):
        print("ARGV_JSON=" + json.dumps(argv, ensure_ascii=False))
    print("resume=normal invocation resumes exact-current stages")
    print("force=--force-stage NAME archives NAME and every later stage")
    return tuple(outcomes)


def _dry_run_commands(
    config: PipelineConfig, audit: InputAudit, runtime: _Runtime
) -> tuple[list[str], ...]:
    fsl = runtime.require_fsl()
    root = config.subject_output
    denoise = root / ".work" / "01_denoise"
    topup_context = _fsl_context(
        config,
        fsl,
        audit,
        topup_dir=root / ".work" / "03_topup",
    )
    bet_context = _fsl_context(
        config,
        fsl,
        audit,
        bet_dir=root / ".work" / "04_bet",
    )
    eddy_context = _fsl_context(
        config,
        fsl,
        audit,
        eddy_dir=root / ".work" / "05_eddy",
    )
    jhu_context = _fsl_context(
        config,
        fsl,
        audit,
        jhu_dir=root / ".work" / "09_jhu_48roi",
    )
    matlab = runtime.require_matlab()
    workers = _pipeline_noddi_workers(config)
    noddi = NODDIContext(
        config=config,
        package_root=_PACKAGE_ROOT,
        stage_dir=config.subject_output / ".work" / "08_noddi",
        eddy_dwi=_paths(config)["eddy_dwi"],
        cleaned_mask=_paths(config)["brain_mask"],
        bvals=config.bvals,
        rotated_bvecs=_paths(config)["rotated_bvecs"],
        matlab=matlab,
        workers=workers,
    )
    commands = [
        [
            str(fsl.bet),
            str(denoise / "raw_mean_b0.nii.gz"),
            str(denoise / "raw_mean_b0_bet"),
            "-R",
            "-f",
            "0.25",
            "-g",
            "0",
            "-m",
        ],
        build_topup_command(topup_context),
        build_topup_mean_command(bet_context),
        build_bet_command(bet_context),
        build_eddy_command(eddy_context),
        build_eddy_quad_command(eddy_context),
        build_prepare_command(noddi),
        *(
            build_worker_command(noddi, worker, workers)
            for worker in range(1, workers + 1)
        ),
        build_merge_command(noddi, workers),
        *build_jhu_commands(jhu_context),
    ]
    return tuple(commands)


def _discover_runtime(config: PipelineConfig) -> _Runtime:
    runtime = _Runtime(config)
    runtime.require_fsl()
    runtime.require_matlab()
    return runtime


def _software_provenance(runtime: _Runtime) -> Mapping[str, str]:
    evidence = dict(_base_software_provenance())
    evidence.update(_fsl_software_provenance(runtime.require_fsl()))
    evidence.update(_matlab_software_provenance(runtime.require_matlab()))
    return MappingProxyType(dict(sorted(evidence.items())))


def _stage_software_provenance(
    runtime: _Runtime, stage_name: str
) -> Mapping[str, str]:
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"unknown stage: {stage_name}")
    evidence = dict(_base_software_provenance())
    if stage_name in _FSL_STAGES:
        evidence.update(_fsl_software_provenance(runtime.require_fsl()))
    if stage_name in _MATLAB_STAGES:
        evidence.update(_matlab_software_provenance(runtime.require_matlab()))
    return MappingProxyType(dict(sorted(evidence.items())))


def _stage_runner(
    config: PipelineConfig, runtime: _Runtime, stage_name: str
) -> StageRunner:
    return StageRunner(
        StageContext(
            config=config,
            package_root=_PACKAGE_ROOT,
            subject_root=config.subject_output,
            software=_stage_software_provenance(runtime, stage_name),
        )
    )


def _fsl_software_provenance(fsl: FSLInstallation) -> Mapping[str, str]:
    material_files = {
        "fsl_topup": fsl.topup,
        "fsl_applytopup": fsl.applytopup,
        "fsl_bet": fsl.bet,
        "fsl_fslmaths": fsl.fslmaths,
        "fsl_eddy": fsl.eddy,
        "fsl_eddy_quad": fsl.eddy_quad,
        "fsl_flirt": fsl.flirt,
        "fsl_fnirt": fsl.fnirt,
        "fsl_invwarp": fsl.invwarp,
        "fsl_applywarp": fsl.applywarp,
        "fsl_b02b0_config": fsl.b02b0_config,
        "fsl_b02b0_no_subsampling_config": fsl.b02b0_no_subsampling_config,
        "fsl_fa_to_standard_config": fsl.fa_to_standard_config,
        "fsl_standard_fa": fsl.standard_fa,
    }
    values = {
        "fsl_eddy": fsl.eddy.name,
    }
    for label, path in material_files.items():
        values[f"{label}_sha256"] = _software_sha256(path, label)
    for relative, path in sorted(fsl.runtime_material_files.items()):
        logical = Path(relative)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise PipelineDependencyError(
                "FSL runtime material identity must be a safe relative path"
            )
        values[f"fsl_runtime:{logical.as_posix()}:sha256"] = _software_sha256(
            path, f"FSL runtime {logical.as_posix()}"
        )
    return MappingProxyType(dict(sorted(values.items())))


def _matlab_software_provenance(
    matlab: MATLABInstallation,
) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "matlab": matlab.version,
            "matlab_executable_sha256": _software_sha256(
                matlab.executable, "matlab_executable"
            ),
            "matlab_mexext": matlab.mexext,
        }
    )


def _software_sha256(path: Path, label: str) -> str:
    descriptor = -1
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise PipelineDependencyError(
                f"required software dependency {label} must be an explicit "
                f"regular file: {path}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise PipelineDependencyError(
                f"required software dependency {label} changed while opening: {path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = path.lstat()
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise PipelineDependencyError(
                f"required software dependency {label} changed while hashing: {path}"
            )
        return digest.hexdigest()
    except PipelineDependencyError:
        raise
    except OSError as error:
        raise PipelineDependencyError(
            f"cannot hash required software dependency {label}: {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _base_software_provenance() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        }
    )


def _safe_manual_review_transition(
    config: PipelineConfig, runner: StageRunner, audit: InputAudit
) -> bool:
    if not config.analysis.ambiguous_qc_reviewed:
        return False
    audit_dir = runner.final_dir("00_input_audit")
    stripe_dir = runner.final_dir("00_pre_denoise_motion_qc")
    if not audit_dir.is_dir() or not stripe_dir.is_dir():
        return False
    if any(
        runner.final_dir(name).exists()
        or runner.final_dir(name).is_symlink()
        or runner.work_dir(name).exists()
        or runner.work_dir(name).is_symlink()
        for name in STAGE_ORDER[2:]
    ):
        return False
    try:
        stored_audit = _read_json(audit_dir / "input_audit.json", "stored input audit")
        decision = _read_json(stripe_dir / "stripe_decision.json", "stripe decision")
        audit_record = StageRecord.from_dict(
            _read_json(
                audit_dir / ".stage_complete.json", "input-audit completion record"
            )
        )
        stripe_record = StageRecord.from_dict(
            _read_json(
                stripe_dir / ".stage_complete.json", "stripe-QC completion record"
            )
        )
    except (OSError, ValueError, StageStateError):
        return False
    if (
        audit_record.stage != "00_input_audit"
        or stripe_record.stage != "00_pre_denoise_motion_qc"
        or audit_record.subject_id != config.subject_id
        or stripe_record.subject_id != config.subject_id
        or not _record_tracks_output(
            audit_record, audit_dir / "input_audit.json", audit_dir
        )
        or not _record_tracks_output(
            stripe_record, stripe_dir / "stripe_decision.json", stripe_dir
        )
    ):
        return False
    raw_hashes = sorted(
        _sha256(path)
        for path in (config.dwi_pa, config.bvals, config.bvecs, config.b0_ap)
    )
    if sorted(str(entry["sha256"]) for entry in audit_record.inputs) != raw_hashes:
        return False
    return (
        stored_audit == audit.to_dict()
        and decision.get("decision") == "HOLD_FOR_REVIEW"
        and decision.get("ambiguous_reviewed") is False
        and decision.get("subject_id") == config.subject_id
    )


def _record_tracks_output(
    record: StageRecord, output: Path, directory: Path
) -> bool:
    try:
        relative = output.relative_to(directory).as_posix()
        metadata = output.lstat()
    except (OSError, ValueError):
        return False
    if not stat.S_ISREG(metadata.st_mode):
        return False
    digest = _sha256(output)
    return any(
        entry["relative_path"] == relative
        and entry["size"] == metadata.st_size
        and entry["sha256"] == digest
        for entry in record.outputs
    )


def _reject_unsafe_existing_state(
    runner: StageRunner, plan: Sequence[StageSpec]
) -> None:
    for spec in plan:
        final = runner.final_dir(spec.name)
        if final.exists() or final.is_symlink():
            if not runner.is_current(spec):
                raise StageStateError(
                    f"Stage {spec.name!r} has stale/noncurrent final state; "
                    f"use a new output_root or --force-stage {spec.name}"
                )
        work = runner.work_dir(spec.name)
        if work.exists() or work.is_symlink():
            if work.is_symlink() or not work.is_dir():
                raise StageStateError(f"Stage {spec.name!r} work state is unsafe")
            try:
                nonempty = any(work.iterdir())
            except OSError as error:
                raise StageStateError(
                    f"cannot inspect Stage {spec.name!r} work directory"
                ) from error
            if nonempty and spec.name != "08_noddi":
                raise StageStateError(
                    f"Stage {spec.name!r} has a partial work directory; "
                    f"rerun with --force-stage {spec.name}"
                )


def _read_qc_decision(directory: Path) -> QCDecision:
    payload = _read_json(directory / "stripe_decision.json", "stripe decision")
    try:
        return QCDecision(
            status=payload["decision"],  # type: ignore[arg-type]
            high_indices=tuple(
                payload["flagged_indices_zero_based"]["high"]  # type: ignore[index]
            ),
            ambiguous_indices=tuple(
                payload["flagged_indices_zero_based"]["ambiguous"]  # type: ignore[index]
            ),
            exit_code=payload["exit_code"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StageStateError("stripe decision record is malformed") from error


def _pipeline_stage_outcome(outcome: StageOutcome) -> PipelineStageOutcome:
    return PipelineStageOutcome(
        outcome.stage, outcome.status, outcome.directory, outcome.record_path
    )


def _raise_preprocess_boundary(error: PreprocessError) -> None:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ExternalCommandError):
            raise PipelineExternalError(str(error)) from error
        current = current.__cause__
    raise PipelineOutputError(str(error)) from error


def _required_outputs(work: Path, names: Sequence[str]) -> tuple[Path, ...]:
    paths = tuple(work / name for name in names)
    _require_regular_files(paths)
    for path in paths:
        if path.suffix == ".json":
            _read_json(path, path.name)
    return paths


def _require_regular_files(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise StageStateError(f"required stage output is missing: {path.name}") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise StageStateError(f"required stage output is not regular: {path.name}")


def _regular_tree_files(
    root: Path, *, require_single_link: bool = False
) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise StageStateError(f"stage output directory is missing or unsafe: {root}")
    files: list[Path] = []
    identities: set[tuple[int, int]] = set()
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise StageStateError(f"stage output contains a symbolic link: {path}")
        if stat.S_ISREG(metadata.st_mode):
            identity = (metadata.st_dev, metadata.st_ino)
            if require_single_link and (
                metadata.st_nlink != 1 or identity in identities
            ):
                raise StageStateError(
                    f"stage output must be a unique single link: {path}"
                )
            identities.add(identity)
            files.append(path)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise StageStateError(f"stage output contains an unsafe entry: {path}")
    return tuple(files)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageStateError(f"cannot read {label}") from error
    if not isinstance(payload, dict):
        raise StageStateError(f"{label} must be a JSON object")
    _finite_json(payload, label)
    return payload


def _finite_json(value: object, label: str) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite_json(child, label)
    elif isinstance(value, list):
        for child in value:
            _finite_json(child, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise StageStateError(f"{label} contains non-finite values")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _finite_json(payload, path.name)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_text(path: Path, label: str) -> np.ndarray:
    try:
        return np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    except (OSError, ValueError) as error:
        raise StageStateError(f"cannot read {label}") from error


def _load_nifti(path: Path) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(path)
    except (OSError, ValueError) as error:
        raise StageStateError(f"cannot read NIfTI output: {path.name}") from error
    if not np.isfinite(image.affine).all():
        raise StageStateError(f"NIfTI output affine is non-finite: {path.name}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise StageStateError(f"cannot hash software file: {path.name}") from error
    return digest.hexdigest()


def _validate_plan_sources(plan: Sequence[StageSpec]) -> None:
    for spec in plan:
        for path in (*spec.source_paths, *spec.resource_paths):
            try:
                metadata = path.lstat()
            except OSError as error:
                raise PipelineDependencyError(
                    f"stage {spec.name} dependency is missing: {path}"
                ) from error
            if not stat.S_ISREG(metadata.st_mode):
                raise PipelineDependencyError(
                    f"stage {spec.name} dependency must be an explicit regular file: {path}"
                )


def _require_config(config: object) -> None:
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a validated PipelineConfig")


def _validate_subject_input_separation(config: PipelineConfig) -> None:
    """Reject identities that subject-stage invalidation could ever move."""
    output = Path(os.path.abspath(os.fspath(config.subject_output)))
    protected = (
        ("raw PA input", config.dwi_pa, PipelineInputError),
        ("b-values input", config.bvals, PipelineInputError),
        ("b-vectors input", config.bvecs, PipelineInputError),
        ("raw AP input", config.b0_ap, PipelineInputError),
        ("configuration file", config.config_path, PipelineInputError),
        *(
            ("package source/resource", path, PipelineDependencyError)
            for path in _static_package_dependencies()
        ),
    )
    for label, raw_path, _error_type in protected:
        protected_path = Path(os.path.abspath(os.fspath(raw_path)))
        try:
            output.relative_to(protected_path)
        except ValueError:
            continue
        raise PipelineInputError(
            f"subject output must not equal, lie inside, or lie below "
            f"{label}: {protected_path}"
        )
    _reject_existing_symlink_components(output, "subject output")
    seen_identities: dict[tuple[int, int], tuple[str, Path]] = {}
    for label, raw_path, error_type in protected:
        path = Path(os.path.abspath(os.fspath(raw_path)))
        _reject_existing_symlink_components(
            path, label, error_type=error_type
        )
        try:
            metadata = path.lstat()
        except OSError as error:
            raise error_type(f"cannot inspect {label}: {path}") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise error_type(
                f"{label} must be an explicit regular file: {path}"
            )
        if metadata.st_nlink != 1:
            raise error_type(
                f"{label} has an unsafe hard-link identity: {path}"
            )
        try:
            path.relative_to(output)
        except ValueError:
            pass
        else:
            raise PipelineInputError(
                f"{label} must not equal or lie inside subject output: {path}"
            )
        identity = (metadata.st_dev, metadata.st_ino)
        previous = seen_identities.get(identity)
        if previous is not None and previous[1] != path:
            raise error_type(
                f"{label} aliases {previous[0]} by filesystem identity"
            )
        seen_identities[identity] = (label, path)
    if output.exists():
        try:
            metadata = output.lstat()
        except OSError as error:
            raise PipelineInputError(
                "cannot inspect subject output identity"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise PipelineInputError("subject output must be a directory")
        if (metadata.st_dev, metadata.st_ino) in seen_identities:
            raise PipelineInputError(
                "subject output aliases a protected input/resource"
            )


def _static_package_dependencies() -> tuple[Path, ...]:
    module_names = (
        "orchestrator",
        "audit",
        "utils",
        "config",
        "stripe_qc",
        "preprocess",
        "fsl",
        "models",
        "noddi",
        "resources",
        "summary",
        "qc",
        "state",
        "report",
    )
    paths = [
        *(_SOURCE_ROOT / f"{name}.py" for name in module_names),
        _ATLAS_IMAGE,
        _ATLAS_XML,
        _ATLAS_PROVENANCE,
        _HENRIQUE_HELPER,
        *(_PACKAGE_ROOT / "scripts" / "matlab").glob("*.m"),
    ]
    for root in (
        _PACKAGE_ROOT / "vendor" / "noddi_toolbox_v1.05",
        _PACKAGE_ROOT / "vendor" / "nifti_matlab",
    ):
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return tuple(dict.fromkeys(sorted(paths)))


def _reject_existing_symlink_components(
    path: Path,
    label: str,
    *,
    error_type: type[Exception] = PipelineInputError,
) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise error_type(
                f"cannot inspect {label}: {current}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise error_type(
                f"{label} contains an unsafe symbolic link"
            )


def _open_directory_chain(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as error:
        raise StageStateError("cannot pin subject output root directory") from error
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise StageStateError(
                    "cannot pin safe subject output directory chain"
                ) from error
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise StageStateError("pinned subject output is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _lock_anchor_root() -> Path:
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    return temporary_root / f"dmri-repro-locks-{os.getuid()}"


def _acquire_subject_lock_anchor(subject_root: Path) -> tuple[int, int]:
    root = _lock_anchor_root()
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as error:
        raise StageStateError("cannot create safe pipeline lock namespace") from error
    try:
        metadata = root.lstat()
    except OSError as error:
        raise StageStateError("cannot inspect pipeline lock namespace") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise StageStateError(
            "pipeline lock namespace must be a private user-owned directory"
        )
    directory_descriptor = _open_directory_chain(root)
    directory_metadata = os.fstat(directory_descriptor)
    if not _directory_path_matches(root, directory_metadata):
        os.close(directory_descriptor)
        raise StageStateError("pipeline lock namespace identity changed")
    lexical = Path(os.path.abspath(os.fspath(subject_root))).as_posix()
    name = hashlib.sha256(lexical.encode("utf-8")).hexdigest() + ".lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    except OSError as error:
        os.close(directory_descriptor)
        raise StageStateError("cannot open pipeline subject lock guard") from error
    try:
        details = os.fstat(descriptor)
        named = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) & 0o077
            or (named.st_dev, named.st_ino) != (details.st_dev, details.st_ino)
        ):
            raise StageStateError(
                "pipeline subject lock guard has an unsafe identity"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StageStateError(
                f"a pipeline is already running for subject "
                f"{Path(subject_root).name}"
            ) from error
    except BaseException:
        os.close(descriptor)
        os.close(directory_descriptor)
        raise
    return directory_descriptor, descriptor


def _unlock_close(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _directory_path_matches(path: Path, expected: os.stat_result) -> bool:
    try:
        observed = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(observed.st_mode) and (
        observed.st_dev,
        observed.st_ino,
    ) == (expected.st_dev, expected.st_ino)


def _create_subject_root(path: Path) -> None:
    parent = path.parent
    if parent.exists() and parent.is_symlink():
        raise StageStateError("output_root must not be a symbolic link")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StageStateError(f"cannot create subject output: {path}") from error
    _require_safe_subject_root(path)


def _require_safe_subject_root(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise StageStateError(f"cannot inspect subject output: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise StageStateError("subject output contains a symbolic-link component")
        if current != absolute and not stat.S_ISDIR(metadata.st_mode):
            raise StageStateError("subject output contains a non-directory component")
    if not path.is_dir():
        raise StageStateError("subject output must be a directory")


__all__ = [
    "STAGE_ORDER",
    "PipelineDependencyError",
    "PipelineExternalError",
    "PipelineInputError",
    "PipelineOutcome",
    "PipelineOutputError",
    "PipelineStageOutcome",
    "build_plan",
    "run_pipeline",
]
