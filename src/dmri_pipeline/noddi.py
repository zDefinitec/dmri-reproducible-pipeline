"""Validated MATLAB discovery and resumable independent NODDI workers."""

from __future__ import annotations

import hashlib
import errno
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError, ImageDataError

from .config import PipelineConfig
from .utils import InputAuditError, normalize_bvecs


_MODEL_NAME = "WatsonSHStickTortIsoV_B0"
_REQUIRED_MAPS = (
    "NODDI_odi.nii",
    "NODDI_ficvf.nii",
    "NODDI_fiso.nii",
    "NODDI_kappa.nii",
    "NODDI_fmin.nii",
    "NODDI_error_code.nii",
    "NODDI_fibredirs_xvec.nii",
    "NODDI_fibredirs_yvec.nii",
    "NODDI_fibredirs_zvec.nii",
)
_MEX_STEMS = ("file2mat", "mat2file", "init")
_QUICK_PROBE_TIMEOUT_SECONDS = 30.0
_MEX_PROBE_TIMEOUT_SECONDS = 300.0
_QUICK_SENTINELS = {
    "version": "__DMRI_MATLAB_VERSION__",
    "mexext": "__DMRI_MEXEXT__",
    "opt_installed": "__DMRI_OPT_INSTALLED__",
    "opt_licensed": "__DMRI_OPT_LICENSED__",
    "mex_configured": "__DMRI_MEX_CONFIGURED__",
}
_MEX_SENTINELS = {
    "mex_works": "__DMRI_MEX_WORKS__",
}
_MEXEXT = re.compile(r"^mex[A-Za-z0-9_]+$")


class MATLABDiscoveryError(RuntimeError):
    """Raised when MATLAB cannot be found or lacks required capabilities."""


class NODDIError(RuntimeError):
    """Raised when NODDI preparation, worker execution, or merge is unsafe."""


class NODDIExternalCommandError(NODDIError):
    """Raised only when a MATLAB process cannot run successfully."""


class CommandResult(Protocol):
    returncode: int


CommandRunner = Callable[[Sequence[str], Path], CommandResult]
ProcessFactory = Callable[..., subprocess.Popen[bytes]]


def _default_command_runner(
    argv: Sequence[str], log_path: Path
) -> subprocess.CompletedProcess[bytes]:
    """Run MATLAB directly with one safe combined runtime log."""
    command = _validated_argv(argv)
    descriptor = _open_log(log_path)
    try:
        with os.fdopen(descriptor, "ab", buffering=0) as log:
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    check=False,
                )
            except (OSError, TypeError, ValueError) as error:
                raise NODDIExternalCommandError(
                    f"MATLAB command could not launch; log={log_path}: {error}"
                ) from error
    except OSError as error:
        raise NODDIError(f"cannot write MATLAB runtime log: {log_path}") from error
    if completed.returncode != 0:
        raise NODDIExternalCommandError(
            f"MATLAB command exited with code {completed.returncode}; "
            f"log={log_path}"
        )
    return completed


@dataclass(frozen=True)
class MATLABInstallation:
    """One probed MATLAB executable and its required capabilities."""

    executable: Path
    version: str
    mexext: str
    optimization_toolbox: bool
    mex_configured: bool

    def __post_init__(self) -> None:
        if not isinstance(self.executable, Path):
            raise MATLABDiscoveryError("MATLAB executable must be a pathlib.Path")
        if not isinstance(self.version, str) or not self.version.strip():
            raise MATLABDiscoveryError("MATLAB version probe was empty")
        if not isinstance(self.mexext, str) or not _MEXEXT.fullmatch(self.mexext):
            raise MATLABDiscoveryError("MATLAB returned an invalid mexext")
        if self.optimization_toolbox is not True:
            raise MATLABDiscoveryError(
                "MATLAB Optimization Toolbox must be installed and licensed"
            )
        if self.mex_configured is not True:
            raise MATLABDiscoveryError(
                "MATLAB MEX is not configured; run mex -setup C during setup"
            )


@dataclass(frozen=True)
class NODDIContext:
    """Immutable paths, software, and injected process boundaries for NODDI."""

    config: PipelineConfig
    package_root: Path
    stage_dir: Path
    eddy_dwi: Path
    cleaned_mask: Path
    bvals: Path
    rotated_bvecs: Path
    matlab: MATLABInstallation
    workers: int
    command_runner: CommandRunner = field(
        default=_default_command_runner, repr=False, compare=False
    )
    process_factory: ProcessFactory = field(
        default=subprocess.Popen, repr=False, compare=False
    )
    poll_interval_seconds: float = field(default=0.05, repr=False, compare=False)
    terminate_grace_seconds: float = field(default=5.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, PipelineConfig):
            raise NODDIError("config must be a PipelineConfig")
        if not isinstance(self.matlab, MATLABInstallation):
            raise NODDIError("matlab must be a MATLABInstallation")
        for name in (
            "package_root",
            "stage_dir",
            "eddy_dwi",
            "cleaned_mask",
            "bvals",
            "rotated_bvecs",
        ):
            path = getattr(self, name)
            if not isinstance(path, Path):
                raise NODDIError(f"{name} must be a pathlib.Path")
            if os.pardir in path.parts:
                raise NODDIError(f"{name} must not contain parent traversal")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise NODDIError("workers must be a positive integer")
        if not callable(self.command_runner) or not callable(self.process_factory):
            raise NODDIError("command and process factories must be callable")
        for name in ("poll_interval_seconds", "terminate_grace_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise NODDIError(f"{name} must be a finite nonnegative number")
        _validate_distinct_upstream_paths(self)


@dataclass(frozen=True)
class NODDIPreparation:
    roi_file: Path
    metadata: Path
    brain_voxel_count: int
    worker_count: int
    resumed: bool


@dataclass(frozen=True)
class NODDIWorkerLaunch:
    worker_count: int
    logs: tuple[Path, ...]


@dataclass(frozen=True)
class NODDIMerge:
    maps: Mapping[str, Path]
    parameters: Path
    metrics: Path
    total_voxels: int
    success_count: int
    error_999_count: int
    other_error_count: int


def discover_matlab(config: PipelineConfig) -> MATLABInstallation:
    """Discover MATLAB from config, environment, then the server PATH.

    Direct library callers may omit ``DMRI_EXPECTED_MATLAB_VERSION`` to run
    capability-only discovery. Public wrappers always export it and therefore
    enforce the exact configured version, including for YAML-selected MATLAB.
    """
    if not isinstance(config, PipelineConfig):
        raise MATLABDiscoveryError("config must be a PipelineConfig")
    configured = config.matlab_executable
    environment_configured = os.environ.get("MATLAB_EXECUTABLE")
    if configured is not None:
        candidate = _normalize_matlab_candidate(configured)
        source = "explicit tools.matlab_executable"
    elif environment_configured:
        candidate = _normalize_matlab_candidate(Path(environment_configured))
        source = "process MATLAB_EXECUTABLE"
    else:
        on_path = shutil.which("matlab")
        if on_path is None:
            raise MATLABDiscoveryError(
                "MATLAB was not found: configure tools.matlab_executable, "
                "set MATLAB_EXECUTABLE, or put matlab on PATH"
            )
        candidate = _normalize_matlab_candidate(Path(on_path))
        source = "matlab on PATH"

    if not _is_executable_file(candidate):
        raise MATLABDiscoveryError(
            f"invalid MATLAB installation from {source}: {candidate}"
        )

    quick_stdout = _run_matlab_probe(
        candidate,
        _matlab_quick_probe_expression(),
        timeout=_QUICK_PROBE_TIMEOUT_SECONDS,
        label="capability",
    )
    values = _parse_probe(quick_stdout, _QUICK_SENTINELS)
    expected_version = os.environ.get("DMRI_EXPECTED_MATLAB_VERSION")
    if expected_version is not None:
        if not expected_version:
            raise MATLABDiscoveryError(
                "DMRI_EXPECTED_MATLAB_VERSION must not be empty"
            )
        if values["version"] != expected_version:
            raise MATLABDiscoveryError(
                "MATLAB version mismatch: expected "
                f"{expected_version}, found {values['version']}"
            )
    if values["mexext"] != "mexa64":
        raise MATLABDiscoveryError("MATLAB mexext must be mexa64")
    if values["opt_installed"] != "1" or values["opt_licensed"] != "1":
        raise MATLABDiscoveryError(
            "MATLAB Optimization Toolbox is not installed and licensed"
        )
    if values["mex_configured"] != "1":
        raise MATLABDiscoveryError(
            "MATLAB MEX is not callable/configured; setup must run mex -setup C"
        )
    mex_stdout = _run_matlab_probe(
        candidate,
        _matlab_mex_probe_expression(),
        timeout=_MEX_PROBE_TIMEOUT_SECONDS,
        label="MEX compile/load/run capability",
    )
    mex_values = _parse_probe(mex_stdout, _MEX_SENTINELS)
    if mex_values["mex_works"] != "1":
        raise MATLABDiscoveryError(
            "MATLAB C MEX compiler could not compile, load, and run a "
            "temporary probe"
        )
    return MATLABInstallation(
        executable=candidate.resolve(strict=True),
        version=values["version"],
        mexext=values["mexext"],
        optimization_toolbox=True,
        mex_configured=True,
    )


def _run_matlab_probe(
    candidate: Path, expression: str, *, timeout: float, label: str
) -> str:
    try:
        completed = subprocess.run(
            (str(candidate), "-batch", expression),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise MATLABDiscoveryError(
            f"MATLAB {label} probe timed out after {timeout:g}s"
        ) from error
    except (OSError, TypeError, ValueError) as error:
        raise MATLABDiscoveryError(
            f"MATLAB {label} probe could not launch: {error}"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise MATLABDiscoveryError(
            f"MATLAB {label} probe exited with exit code "
            f"{completed.returncode}{suffix}"
        )
    return completed.stdout


def choose_noddi_workers(
    cpu_count: int, memory_gib: Real, configured: int | str
) -> int:
    """Select bounded independent MATLAB processes with CPU/memory reserves."""
    if (
        isinstance(cpu_count, bool)
        or not isinstance(cpu_count, int)
        or cpu_count < 1
    ):
        raise NODDIError("cpu_count must be a positive integer")
    if (
        isinstance(memory_gib, bool)
        or not isinstance(memory_gib, Real)
        or not math.isfinite(float(memory_gib))
        or float(memory_gib) <= 0
    ):
        raise NODDIError("memory_gib must be a positive finite number")
    if configured == "auto":
        return min(
            8,
            max(1, cpu_count - 2),
            max(1, math.floor(float(memory_gib) / 8.0)),
        )
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
    ):
        raise NODDIError("configured workers must be 'auto' or a positive integer")
    if configured > cpu_count:
        raise NODDIError("configured workers must not exceed available CPUs")
    return configured


def build_prepare_command(context: NODDIContext) -> list[str]:
    """Build compilation plus streaming-ROI MATLAB argv as a fresh list."""
    context = _require_context(context)
    stage = context.stage_dir
    compat = stage / "nifti_matlab" / "matlab"
    private = compat / "@file_array" / "private"
    scripts = context.package_root / "scripts" / "matlab"
    noddi = context.package_root / "vendor" / "noddi_toolbox_v1.05"
    statements = [
        f"compat='{_matlab_quote(compat)}'",
        f"privatePath='{_matlab_quote(private)}'",
    ]
    for stem in _MEX_STEMS:
        binary = private / f"{stem}.{context.matlab.mexext}"
        if not os.path.lexists(binary):
            statements.append(
                "mex('-silent','-outdir',privatePath,'-output',"
                f"'{stem}',fullfile(privatePath,'{stem}.c'))"
            )
    statements.extend(
        [
            f"addpath('{_matlab_quote(scripts)}','-begin')",
            f"addpath('{_matlab_quote(compat)}','-begin')",
            f"addpath('{_matlab_quote(noddi / 'fitting')}','-end')",
            f"addpath(genpath('{_matlab_quote(noddi / 'models')}'),'-end')",
        ]
    )
    if not os.path.lexists(stage / "NODDI_roi.mat"):
        statements.append(
            "CreateROI("
            f"'{_matlab_quote(stage / 'eddy_dwi.nii')}',"
            f"'{_matlab_quote(stage / 'cleaned_mask.nii')}',"
            f"'{_matlab_quote(stage / 'NODDI_roi.mat')}')"
        )
    return [str(context.matlab.executable), "-batch", ";".join(statements)]


def build_worker_command(
    context: NODDIContext, worker: int, workers: int
) -> list[str]:
    """Build one shell-free MATLAB worker argv with safe literal escaping."""
    context = _require_context(context)
    _validate_worker_pair(worker, workers)
    scripts = context.package_root / "scripts" / "matlab"
    expression = (
        f"addpath('{_matlab_quote(scripts)}','-begin');"
        f"run_noddi_worker('{_matlab_quote(context.stage_dir)}',"
        f"{worker},{workers})"
    )
    return [str(context.matlab.executable), "-batch", expression]


def build_merge_command(context: NODDIContext, workers: int) -> list[str]:
    """Build deterministic merge argv."""
    context = _require_context(context)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise NODDIError("workers must be a positive integer")
    scripts = context.package_root / "scripts" / "matlab"
    expression = (
        f"addpath('{_matlab_quote(scripts)}','-begin');"
        f"merge_noddi_workers('{_matlab_quote(context.stage_dir)}',{workers})"
    )
    return [str(context.matlab.executable), "-batch", expression]


def prepare_noddi(context: NODDIContext) -> NODDIPreparation:
    """Validate inputs, materialize compatibility sources, compile, and make ROI."""
    context = _require_context(context)
    package_sources = _validate_package_sources(context)
    inputs = _load_preparation_inputs(context)
    source_hashes = _source_hashes(context, package_sources)
    metadata_path = context.stage_dir / "noddi_prepare.json"
    state_path = context.stage_dir / "noddi_prepare_state.json"

    if os.path.lexists(metadata_path):
        payload = _read_json_regular(metadata_path, "NODDI preparation metadata")
        _validate_resume_metadata(context, payload, inputs, source_hashes)
        return NODDIPreparation(
            roi_file=context.stage_dir / "NODDI_roi.mat",
            metadata=metadata_path,
            brain_voxel_count=inputs["brain_voxel_count"],
            worker_count=min(context.workers, inputs["brain_voxel_count"]),
            resumed=True,
        )

    _ensure_safe_directory(context.stage_dir, create=True, label="NODDI stage")
    if os.path.lexists(state_path):
        state = _read_json_regular(state_path, "partial NODDI preparation state")
        _validate_partial_state(context, state, inputs, source_hashes)
    else:
        destinations = _preparation_destinations(context)
        existing = [path for path in destinations if os.path.lexists(path)]
        if existing:
            raise NODDIError(
                "unsafe pre-existing NODDI preparation destination without valid "
                "resume metadata: " + ", ".join(path.name for path in existing)
            )

        _write_uncompressed_copy(context.eddy_dwi, context.stage_dir / "eddy_dwi.nii")
        _write_mask_copy(context.cleaned_mask, context.stage_dir / "cleaned_mask.nii")
        rounded = _round_shells_half_away(np.asarray(inputs["bvals"], dtype=float))
        _atomic_savetxt(
            context.stage_dir / "bvals_rounded.txt", rounded.reshape(1, -1), "%d"
        )
        _atomic_savetxt(
            context.stage_dir / "eddy_rotated_bvecs.txt",
            np.asarray(inputs["bvecs"], dtype=float),
            "%.12g",
        )
        _copy_source_tree(
            context.package_root / "vendor" / "nifti_matlab",
            context.stage_dir / "nifti_matlab",
        )
        _atomic_json(
            state_path,
            _partial_state_payload(context, inputs, source_hashes),
        )

    before = _upstream_hashes(context)
    if not _preparation_runtime_outputs_complete(context, inputs):
        context.command_runner(
            build_prepare_command(context), context.stage_dir / "noddi_prepare.log"
        )
    after = _upstream_hashes(context)
    if before != after:
        raise NODDIError("upstream NODDI inputs changed during preparation")

    _validate_compiled_mex(context)
    _validate_roi(
        context.stage_dir / "NODDI_roi.mat",
        brain_voxels=int(inputs["brain_voxel_count"]),
        volumes=int(inputs["dwi_shape"][-1]),
        spatial_shape=tuple(inputs["dwi_shape"][:3]),
    )
    worker_count = min(context.workers, int(inputs["brain_voxel_count"]))
    payload = {
        "schema_version": 1,
        "dwi_shape": list(inputs["dwi_shape"]),
        "mask_shape": list(inputs["mask_shape"]),
        "shell_counts": {
            str(shell): int(count)
            for shell, count in sorted(inputs["shell_counts"].items())
        },
        "brain_voxel_count": int(inputs["brain_voxel_count"]),
        "chosen_worker_count": worker_count,
        "matlab_version": context.matlab.version,
        "mexext": context.matlab.mexext,
        "model_name": _MODEL_NAME,
        "source_hashes": dict(sorted(source_hashes.items())),
        "artifacts": _preparation_artifact_hashes(context),
    }
    _atomic_json(metadata_path, payload)
    state_path.unlink(missing_ok=True)
    return NODDIPreparation(
        context.stage_dir / "NODDI_roi.mat",
        metadata_path,
        int(inputs["brain_voxel_count"]),
        worker_count,
        False,
    )


def launch_noddi_workers(context: NODDIContext) -> NODDIWorkerLaunch:
    """Launch independent MATLAB processes and fail peers closed."""
    context = _require_context(context)
    payload, _ = _validate_current_preparation(context)
    roi_voxels = _positive_int(payload.get("brain_voxel_count"), "brain voxel count")
    workers = min(context.workers, roi_voxels)
    logs = tuple(
        context.stage_dir / f"worker_{worker:02d}.log"
        for worker in range(1, workers + 1)
    )
    for worker in range(1, workers + 1):
        for suffix in ("checkpoint.mat", "final.mat"):
            path = context.stage_dir / f"worker_{worker:02d}_{suffix}"
            if os.path.lexists(path):
                _require_regular_single_link(path, "NODDI worker state")
    processes: list[tuple[int, object, object]] = []
    try:
        for worker, log_path in enumerate(logs, start=1):
            descriptor = _open_log(log_path)
            log_handle = os.fdopen(descriptor, "ab", buffering=0)
            try:
                process = context.process_factory(
                    build_worker_command(context, worker, workers),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
            except Exception:
                log_handle.close()
                raise
            processes.append((worker, process, log_handle))
    except Exception as error:
        _stop_workers(processes, context.terminate_grace_seconds)
        _close_worker_logs(processes)
        raise NODDIExternalCommandError(
            f"could not spawn NODDI worker {len(processes) + 1}; "
            f"log={logs[len(processes)]}; existing checkpoints were preserved; "
            f"resume with {_resume_instruction(context)}"
        ) from error

    try:
        pending = {worker for worker, _, _ in processes}
        while pending:
            for worker, process, _ in processes:
                if worker not in pending:
                    continue
                try:
                    returncode = process.poll()
                except Exception as error:
                    _stop_workers(processes, context.terminate_grace_seconds)
                    raise NODDIExternalCommandError(
                        f"NODDI worker {worker} poll failed; "
                        f"log={logs[worker - 1]}; resume with "
                        f"{_resume_instruction(context)}"
                    ) from error
                if returncode is None:
                    continue
                pending.remove(worker)
                if returncode != 0:
                    _stop_workers(
                        [
                            entry
                            for entry in processes
                            if entry[0] in pending
                        ],
                        context.terminate_grace_seconds,
                    )
                    raise NODDIExternalCommandError(
                        f"NODDI worker {worker} exited with code {returncode}; "
                        f"log={logs[worker - 1]}; checkpoints/finals preserved; "
                        f"resume with {_resume_instruction(context)}"
                    )
                final = context.stage_dir / f"worker_{worker:02d}_final.mat"
                if not _validate_worker_final(
                    final, worker, workers, roi_voxels
                ):
                    _stop_workers(
                        [
                            entry
                            for entry in processes
                            if entry[0] in pending
                        ],
                        context.terminate_grace_seconds,
                    )
                    raise NODDIError(
                        f"NODDI worker {worker} exited zero but did not produce "
                        f"a safe final file; log={logs[worker - 1]}; resume with "
                        f"{_resume_instruction(context)}"
                    )
            if pending and context.poll_interval_seconds:
                time.sleep(context.poll_interval_seconds)
    finally:
        _close_worker_logs(processes)
    return NODDIWorkerLaunch(workers, logs)


def merge_noddi(context: NODDIContext) -> NODDIMerge:
    """Run MATLAB merge then independently validate maps and metrics."""
    context = _require_context(context)
    payload, _ = _validate_current_preparation(context)
    _validate_merge_destinations(context)
    workers = min(
        context.workers,
        int(payload.get("brain_voxel_count", context.workers)),
    )
    context.command_runner(
        build_merge_command(context, workers), context.stage_dir / "merge_noddi.log"
    )
    post_merge_payload, _ = _validate_current_preparation(context)
    if post_merge_payload != payload:
        raise NODDIError(
            "NODDI preparation metadata changed during the MATLAB merge command"
        )
    mask_image = _load_image(context.cleaned_mask, "cleaned mask")
    mask = _stream_mask(mask_image)
    maps = {
        name: context.stage_dir / name
        for name in _REQUIRED_MAPS
    }
    arrays = {
        name: _validate_merge_map(path, mask_image)
        for name, path in maps.items()
    }
    error_values = arrays["NODDI_error_code.nii"]
    if not np.allclose(error_values, np.rint(error_values), atol=0, rtol=0):
        raise NODDIError("NODDI error-code map must contain integral values")
    success = mask & (error_values == 0)
    for name in ("NODDI_odi.nii", "NODDI_ficvf.nii", "NODDI_fiso.nii"):
        values = arrays[name][success]
        if not np.isfinite(values).all():
            raise NODDIError(f"{name} must be finite for successful mask voxels")
        if np.any((values < 0) | (values > 1)):
            raise NODDIError(f"{name} must be within [0, 1] for successful voxels")
    for name in (
        "NODDI_kappa.nii",
        "NODDI_fmin.nii",
        "NODDI_fibredirs_xvec.nii",
        "NODDI_fibredirs_yvec.nii",
        "NODDI_fibredirs_zvec.nii",
    ):
        if not np.isfinite(arrays[name][success]).all():
            raise NODDIError(f"{name} must be finite for successful mask voxels")
    fibre_norms = np.sqrt(
        arrays["NODDI_fibredirs_xvec.nii"][success] ** 2
        + arrays["NODDI_fibredirs_yvec.nii"][success] ** 2
        + arrays["NODDI_fibredirs_zvec.nii"][success] ** 2
    )
    if np.any(~np.isclose(fibre_norms, 1.0, atol=1e-4, rtol=0)):
        raise NODDIError(
            "NODDI fibre directions must have unit length for successful voxels"
        )

    params = context.stage_dir / "NODDI_params.mat"
    if not _safe_regular_file(params):
        raise NODDIError("NODDI parameter MAT file is missing or unsafe")
    mat_errors, mlps_shape = _load_mat_arrays(params)
    roi_indices = _load_roi_indices(context.stage_dir / "NODDI_roi.mat")
    if mat_errors.shape != (roi_indices.shape[0],):
        raise NODDIError("NODDI parameter MAT error-code length is inconsistent")
    if sorted(mlps_shape) != sorted((roi_indices.shape[0], 8)):
        raise NODDIError("NODDI parameter MAT has the wrong model parameter width")
    if np.any(roi_indices > np.asarray(mask.shape, dtype=np.int64)):
        raise NODDIError("NODDI ROI idx coordinate is outside the output map")
    mapped_errors = np.asarray(
        [
            error_values[int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1]
            for row in roi_indices
        ],
        dtype=float,
    )
    if not np.array_equal(mat_errors, mapped_errors):
        raise NODDIError(
            "NODDI parameter MAT error codes do not match the error-code map"
        )
    metrics_path = context.stage_dir / "noddi_metrics.json"
    metrics = _read_json_regular(metrics_path, "NODDI metrics")
    total = int(np.count_nonzero(mask))
    success_count = int(np.count_nonzero(mask & (error_values == 0)))
    error_999 = int(np.count_nonzero(mask & (error_values == 999)))
    other = total - success_count - error_999
    expected = {
        "total_voxels": total,
        "success_count": success_count,
        "error_999_count": error_999,
        "other_error_count": other,
        "worker_count": workers,
        "model_name": _MODEL_NAME,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise NODDIError(f"NODDI metrics JSON is inconsistent for {key}")
    if success_count + error_999 + other != total or other < 0:
        raise NODDIError("NODDI error counts do not partition the ROI")
    listed = metrics.get("parameter_maps")
    if not isinstance(listed, list) or not set(_REQUIRED_MAPS).issubset(listed):
        raise NODDIError("NODDI metrics JSON parameter map list is incomplete")
    finite_objectives = arrays["NODDI_fmin.nii"][mask]
    finite_objectives = finite_objectives[np.isfinite(finite_objectives)]
    objective_expected: dict[str, object] = {
        "objective_finite_count": int(finite_objectives.size),
        "objective_min": (
            float(np.min(finite_objectives)) if finite_objectives.size else None
        ),
        "objective_max": (
            float(np.max(finite_objectives)) if finite_objectives.size else None
        ),
        "objective_mean": (
            float(np.mean(finite_objectives)) if finite_objectives.size else None
        ),
    }
    for key, value in objective_expected.items():
        recorded = metrics.get(key)
        if value is None:
            if recorded is not None:
                raise NODDIError(f"NODDI metrics JSON is inconsistent for {key}")
        elif not isinstance(recorded, Real) or not math.isclose(
            float(recorded), float(value), rel_tol=1e-7, abs_tol=1e-12
        ):
            raise NODDIError(f"NODDI metrics JSON is inconsistent for {key}")
    return NODDIMerge(
        MappingProxyType(maps),
        params,
        metrics_path,
        total,
        success_count,
        error_999,
        other,
    )


def _normalize_matlab_candidate(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.suffix == ".app" or candidate.is_dir():
        candidate = candidate / "bin" / "matlab"
    return candidate.resolve(strict=False)


def _is_executable_file(path: Path) -> bool:
    try:
        metadata = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and os.access(path, os.X_OK)


def _matlab_quick_probe_expression() -> str:
    return (
        "fprintf('__DMRI_MATLAB_VERSION__=%s\\n',version);"
        "fprintf('__DMRI_MEXEXT__=%s\\n',mexext);"
        "v=ver('optim');"
        "fprintf('__DMRI_OPT_INSTALLED__=%d\\n',~isempty(v));"
        "fprintf('__DMRI_OPT_LICENSED__=%d\\n',license('test','Optimization_Toolbox'));"
        "try,c=mex.getCompilerConfigurations('C','Selected');ok=~isempty(c);"
        "catch,ok=false;end;"
        "fprintf('__DMRI_MEX_CONFIGURED__=%d\\n',ok)"
    )


def _matlab_mex_probe_expression() -> str:
    return (
        "d=tempname;mkdir(d);cleanup_dir=onCleanup(@()rmdir(d,'s'));"
        "src=fullfile(d,'dmri_mex_probe.c');q=char(34);"
        "code=['#include ' q 'mex.h' q newline "
        "'void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){plhs[0]=mxCreateDoubleScalar(42.0);}' newline];"
        "fid=fopen(src,'w');assert(fid>=0);fwrite(fid,code);fclose(fid);"
        "mex_works=false;try,mex('-silent','-outdir',d,src);"
        "addpath(d);cleanup_path=onCleanup(@()rmpath(d));"
        "clear dmri_mex_probe;y=dmri_mex_probe();"
        "mex_works=isscalar(y)&&isfinite(y)&&y==42;"
        "clear dmri_mex_probe cleanup_path;"
        "catch ME,disp(getReport(ME,'extended','hyperlinks','off'));"
        "clear dmri_mex_probe;if exist('cleanup_path','var'),clear cleanup_path;end;end;"
        "fprintf('__DMRI_MEX_WORKS__=%d\\n',mex_works);clear cleanup_dir"
    )


def _parse_probe(
    stdout: str, sentinels: Mapping[str, str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, sentinel in sentinels.items():
        matches = re.findall(
            rf"(?m)^{re.escape(sentinel)}=(.*?)\r?$", stdout or ""
        )
        if len(matches) != 1 or not matches[0].strip():
            raise MATLABDiscoveryError(
                f"MATLAB probe sentinel {sentinel} is missing, duplicate, or malformed"
            )
        values[key] = matches[0].strip()
    if "mexext" in values and not _MEXEXT.fullmatch(values["mexext"]):
        raise MATLABDiscoveryError("MATLAB probe returned malformed mexext sentinel")
    for key in ("opt_installed", "opt_licensed", "mex_configured", "mex_works"):
        if key in values and values[key] not in {"0", "1"}:
            raise MATLABDiscoveryError(
                f"MATLAB probe sentinel {sentinels[key]} must be 0 or 1"
            )
    return values


def _matlab_quote(path: Path) -> str:
    value = str(path)
    if "\x00" in value or "\n" in value or "\r" in value:
        raise NODDIError("MATLAB paths must not contain NUL or newlines")
    return value.replace("'", "''")


def _validate_worker_pair(worker: object, workers: object) -> None:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or isinstance(worker, bool)
        or not isinstance(worker, int)
        or not 1 <= worker <= workers
    ):
        raise NODDIError("worker index must be from 1 through worker count")


def _require_context(context: object) -> NODDIContext:
    if not isinstance(context, NODDIContext):
        raise NODDIError("context must be a NODDIContext")
    _validate_distinct_upstream_paths(context)
    return context


def _validate_distinct_upstream_paths(context: NODDIContext) -> None:
    paths = (
        context.eddy_dwi,
        context.cleaned_mask,
        context.bvals,
        context.rotated_bvecs,
    )
    lexical = [Path(os.path.abspath(path)) for path in paths]
    if len(set(lexical)) != len(lexical):
        raise NODDIError("NODDI upstream paths must be distinct")
    resolved = [_resolve(path) for path in lexical]
    if len(set(resolved)) != len(resolved):
        raise NODDIError("NODDI upstream paths must not alias by symbolic link")
    identities: dict[tuple[int, int], Path] = {}
    for path in lexical:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise NODDIError(f"cannot inspect NODDI upstream input: {path}") from error
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in identities:
            raise NODDIError(
                f"NODDI upstream inputs must not be hard-linked aliases: "
                f"{identities[identity]} and {path}"
            )
        identities[identity] = path
    stage = _resolve(context.stage_dir)
    for upstream in resolved:
        if stage == upstream:
            raise NODDIError("NODDI stage directory must not alias an upstream input")
        try:
            upstream.relative_to(stage)
        except ValueError:
            pass
        else:
            raise NODDIError("NODDI stage directory must not contain upstream inputs")


def _validate_package_sources(context: NODDIContext) -> tuple[Path, ...]:
    root = _resolve(context.package_root)
    expected = (
        root / "scripts" / "matlab" / "CreateROI.m",
        root / "scripts" / "matlab" / "run_noddi_worker.m",
        root / "scripts" / "matlab" / "merge_noddi_workers.m",
    )
    vendor_roots = (
        root / "vendor" / "noddi_toolbox_v1.05",
        root / "vendor" / "nifti_matlab",
    )
    files = list(expected)
    for vendor in vendor_roots:
        _validate_source_tree(vendor)
        files.extend(path for path in vendor.rglob("*") if path.is_file())
    for path in expected:
        _require_regular_single_link(path, "package MATLAB source")
    return tuple(sorted(files))


def _validate_source_tree(root: Path) -> None:
    _ensure_safe_directory(root, create=False, label="vendor source tree")
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise NODDIError(f"vendor source tree contains a symbolic link: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise NODDIError(f"vendor source entry is unsafe: {path}")
        lower = path.name.lower()
        if (
            ".mex" in lower
            or lower.endswith((".class", ".o", ".obj", ".un~", ".pyc", ".log"))
            or "__pycache__" in path.parts
            or lower == ".ds_store"
        ):
            raise NODDIError(f"vendor source tree contains forbidden binary/cache: {path}")


def _load_preparation_inputs(context: NODDIContext) -> dict[str, object]:
    dwi = _load_image(context.eddy_dwi, "EDDY DWI")
    mask_image = _load_image(context.cleaned_mask, "cleaned mask")
    if len(dwi.shape) != 4:
        raise NODDIError("EDDY DWI must be 4D")
    if len(mask_image.shape) != 3:
        raise NODDIError("cleaned mask must be 3D")
    if tuple(dwi.shape[:3]) != tuple(mask_image.shape):
        raise NODDIError("EDDY DWI and cleaned mask must have matching spatial shape")
    if not np.allclose(dwi.affine, mask_image.affine, atol=1e-5, rtol=0):
        raise NODDIError("EDDY DWI and cleaned mask must use the same affine/grid")
    if not np.isfinite(dwi.affine).all() or not np.isfinite(mask_image.affine).all():
        raise NODDIError("NODDI input affines must be finite")
    for volume in range(int(dwi.shape[3])):
        try:
            values = np.asanyarray(dwi.dataobj[..., volume])
        except (OSError, ValueError, TypeError, ImageDataError) as error:
            raise NODDIError(f"cannot stream EDDY DWI volume {volume}") from error
        if not np.isfinite(values).all():
            raise NODDIError(f"EDDY DWI volume {volume} contains non-finite data")
    mask = _stream_mask(mask_image)
    brain_voxels = int(np.count_nonzero(mask))
    if brain_voxels < 1:
        raise NODDIError("cleaned mask must contain at least one voxel")
    bvals = _load_text(context.bvals, "b-values").reshape(-1)
    if bvals.size != dwi.shape[3]:
        raise NODDIError("b-value count must equal EDDY DWI volume count")
    if not np.isfinite(bvals).all() or np.any(bvals < 0):
        raise NODDIError("b-values must be finite and nonnegative")
    try:
        bvecs = normalize_bvecs(
            _load_text(context.rotated_bvecs, "EDDY-rotated b-vectors"),
            int(dwi.shape[3]),
        )
    except InputAuditError as error:
        raise NODDIError(str(error)) from error
    norms = np.linalg.norm(bvecs, axis=0)
    b0 = bvals < 50
    if not b0.any():
        raise NODDIError("NODDI inputs require at least one b0")
    if np.count_nonzero(~b0) < 1:
        raise NODDIError("NODDI inputs require at least one non-b0 diffusion volume")
    accepted_b0 = (norms[b0] < 0.1) | (
        (norms[b0] >= 0.95) & (norms[b0] <= 1.05)
    )
    if not accepted_b0.all():
        raise NODDIError("b0 rotated vectors must be near-zero or unit length")
    diffusion = ~b0
    if np.any((norms[diffusion] < 0.95) | (norms[diffusion] > 1.05)):
        raise NODDIError("non-b0 EDDY-rotated vectors must have unit length")
    shells = _round_shells_half_away(bvals)
    return {
        "dwi_shape": tuple(int(value) for value in dwi.shape),
        "mask_shape": tuple(int(value) for value in mask_image.shape),
        "bvals": bvals,
        "bvecs": bvecs,
        "brain_voxel_count": brain_voxels,
        "shell_counts": {
            int(shell): int(np.count_nonzero(shells == shell))
            for shell in np.unique(shells)
        },
    }


def _round_shells_half_away(bvals: np.ndarray) -> np.ndarray:
    values = np.asarray(bvals, dtype=float)
    rounded = np.floor(values / 100.0 + 0.5).astype(np.int64) * 100
    return np.where(values < 50.0, 0, rounded)


def _write_uncompressed_copy(source: Path, destination: Path) -> None:
    image = _load_image(source, "EDDY DWI")
    _require_absent_safe(destination)
    try:
        nib.save(image, destination)
    except (OSError, HeaderDataError, ImageDataError, ValueError) as error:
        raise NODDIError(f"cannot write uncompressed EDDY NIfTI: {destination}") from error


def _write_mask_copy(source: Path, destination: Path) -> None:
    image = _load_image(source, "cleaned mask")
    mask = _stream_mask(image).astype(np.uint8)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    _require_absent_safe(destination)
    try:
        nib.save(nib.Nifti1Image(mask, image.affine, header), destination)
    except (OSError, HeaderDataError, ImageDataError, ValueError) as error:
        raise NODDIError(f"cannot write uncompressed cleaned mask: {destination}") from error


def _copy_source_tree(source: Path, destination: Path) -> None:
    _validate_source_tree(source)
    _require_absent_safe(destination)
    try:
        shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2)
    except OSError as error:
        raise NODDIError(f"cannot copy NIfTI MATLAB source tree: {error}") from error
    _validate_source_tree(destination)


def _validate_compiled_mex(context: NODDIContext) -> None:
    private = (
        context.stage_dir
        / "nifti_matlab"
        / "matlab"
        / "@file_array"
        / "private"
    )
    for stem in _MEX_STEMS:
        path = private / f"{stem}.{context.matlab.mexext}"
        _require_regular_single_link(path, "compiled NIfTI MEX")
        if path.stat().st_size < 1:
            raise NODDIError(f"compiled NIfTI MEX is empty: {path}")


def _validate_roi(
    path: Path,
    *,
    brain_voxels: int,
    volumes: int,
    spatial_shape: tuple[int, ...],
) -> None:
    _require_regular_single_link(path, "NODDI ROI")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - transitive runtime dependency
        raise NODDIError("h5py is required to validate MATLAB v7.3 ROI files") from error
    try:
        with h5py.File(path, "r") as handle:
            if not {"roi", "idx", "mask"}.issubset(handle.keys()):
                raise NODDIError("NODDI ROI must contain roi, idx, and mask")
            roi_shape = tuple(int(value) for value in handle["roi"].shape)
            idx_shape = tuple(int(value) for value in handle["idx"].shape)
            mask_shape = tuple(int(value) for value in handle["mask"].shape)
            if sorted(roi_shape) != sorted((brain_voxels, volumes)):
                raise NODDIError("NODDI ROI dimensions are inconsistent")
            if sorted(idx_shape) != sorted((brain_voxels, 3)):
                raise NODDIError("NODDI ROI idx dimensions are inconsistent")
            if tuple(mask_shape) not in (tuple(spatial_shape), tuple(reversed(spatial_shape))):
                raise NODDIError("NODDI ROI mask dimensions are inconsistent")
            indices = np.asarray(handle["idx"], dtype=float)
            mask_values = np.asarray(handle["mask"])
            if not _hdf5_dataset_is_finite(handle["roi"]):
                raise NODDIError("NODDI ROI signal values must be finite")
    except OSError as error:
        raise NODDIError("NODDI ROI is not a readable MATLAB v7.3 file") from error
    if indices.shape[1] == 3:
        indices = indices
    elif indices.shape[0] == 3:
        indices = indices.T
    else:  # guarded by sorted shape, retained for clarity
        raise NODDIError("NODDI ROI idx must have three coordinate columns")
    if (
        not np.isfinite(indices).all()
        or not np.array_equal(indices, np.rint(indices))
        or np.any(indices < 1)
    ):
        raise NODDIError("NODDI ROI idx must contain positive integer coordinates")
    indices = np.asarray(indices, dtype=np.int64)
    if np.unique(indices, axis=0).shape[0] != brain_voxels:
        raise NODDIError("NODDI ROI idx coordinates must be unique")
    if np.any(indices > np.asarray(spatial_shape, dtype=np.int64)):
        raise NODDIError("NODDI ROI idx coordinate is outside the mask bounds")
    mask_candidates = [mask_values]
    if mask_values.ndim == 3:
        mask_candidates.append(mask_values.transpose(2, 1, 0))
    correspondence = False
    for candidate in mask_candidates:
        if tuple(candidate.shape) != tuple(spatial_shape):
            continue
        labels = np.asarray(
            [
                candidate[row[0] - 1, row[1] - 1, row[2] - 1]
                for row in indices
            ]
        )
        if (
            np.array_equal(labels, np.arange(1, brain_voxels + 1))
            and np.count_nonzero(candidate) == brain_voxels
        ):
            correspondence = True
            break
    if not correspondence:
        raise NODDIError("NODDI ROI mask and idx correspondence is inconsistent")


def _hdf5_dataset_is_finite(dataset: object) -> bool:
    shape = tuple(int(value) for value in dataset.shape)
    if not shape:
        return False
    step = max(1, min(shape[0], 1024))
    for start in range(0, shape[0], step):
        selection = (slice(start, min(start + step, shape[0])),) + (
            slice(None),
        ) * (len(shape) - 1)
        if not np.isfinite(np.asarray(dataset[selection])).all():
            return False
    return True


def _validate_resume_metadata(
    context: NODDIContext,
    payload: Mapping[str, object],
    inputs: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> None:
    expected = {
        "schema_version": 1,
        "dwi_shape": list(inputs["dwi_shape"]),
        "mask_shape": list(inputs["mask_shape"]),
        "brain_voxel_count": inputs["brain_voxel_count"],
        "chosen_worker_count": min(context.workers, int(inputs["brain_voxel_count"])),
        "matlab_version": context.matlab.version,
        "mexext": context.matlab.mexext,
        "model_name": _MODEL_NAME,
        "source_hashes": dict(sorted(source_hashes.items())),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise NODDIError(f"stale NODDI preparation metadata: {key} changed")
    shell_counts = {
        str(shell): int(count)
        for shell, count in sorted(inputs["shell_counts"].items())
    }
    if payload.get("shell_counts") != shell_counts:
        raise NODDIError("stale NODDI preparation metadata: shell counts changed")
    _validate_compiled_mex(context)
    _validate_roi(
        context.stage_dir / "NODDI_roi.mat",
        brain_voxels=int(inputs["brain_voxel_count"]),
        volumes=int(inputs["dwi_shape"][-1]),
        spatial_shape=tuple(inputs["dwi_shape"][:3]),
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise NODDIError("stale NODDI preparation metadata: artifacts missing")
    actual = _preparation_artifact_hashes(context)
    if dict(artifacts) != actual:
        raise NODDIError("stale NODDI preparation metadata: artifact hash mismatch")


def _validate_current_preparation(
    context: NODDIContext,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = _read_json_regular(
        context.stage_dir / "noddi_prepare.json", "NODDI preparation metadata"
    )
    inputs = _load_preparation_inputs(context)
    package_sources = _validate_package_sources(context)
    _validate_resume_metadata(
        context,
        payload,
        inputs,
        _source_hashes(context, package_sources),
    )
    return payload, inputs


def _source_hashes(
    context: NODDIContext, package_sources: Sequence[Path]
) -> dict[str, str]:
    result = _upstream_hashes(context)
    root = context.package_root.resolve(strict=False)
    for path in package_sources:
        try:
            relative = path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise NODDIError(f"package source escapes package root: {path}") from error
        result[f"package:{relative.as_posix()}"] = _sha256(path)
    return result


def _upstream_hashes(context: NODDIContext) -> dict[str, str]:
    return {
        "eddy_dwi": _sha256(context.eddy_dwi),
        "cleaned_mask": _sha256(context.cleaned_mask),
        "bvals": _sha256(context.bvals),
        "eddy_rotated_bvecs": _sha256(context.rotated_bvecs),
    }


def _preparation_artifact_hashes(context: NODDIContext) -> dict[str, str]:
    private = (
        context.stage_dir
        / "nifti_matlab"
        / "matlab"
        / "@file_array"
        / "private"
    )
    paths = (
        context.stage_dir / "eddy_dwi.nii",
        context.stage_dir / "cleaned_mask.nii",
        context.stage_dir / "bvals_rounded.txt",
        context.stage_dir / "eddy_rotated_bvecs.txt",
        context.stage_dir / "NODDI_roi.mat",
        *(private / f"{stem}.{context.matlab.mexext}" for stem in _MEX_STEMS),
    )
    result = {
        path.relative_to(context.stage_dir).as_posix(): _sha256(path)
        for path in paths
    }
    result["nifti_matlab_tree_sha256"] = _tree_sha256(
        context.stage_dir / "nifti_matlab",
        allowed_mexext=context.matlab.mexext,
    )
    return result


def _preparation_destinations(context: NODDIContext) -> tuple[Path, ...]:
    return (
        context.stage_dir / "eddy_dwi.nii",
        context.stage_dir / "cleaned_mask.nii",
        context.stage_dir / "bvals_rounded.txt",
        context.stage_dir / "eddy_rotated_bvecs.txt",
        context.stage_dir / "NODDI_roi.mat",
        context.stage_dir / "nifti_matlab",
        context.stage_dir / "noddi_prepare.log",
    )


def _partial_state_payload(
    context: NODDIContext,
    inputs: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dwi_shape": list(inputs["dwi_shape"]),
        "mask_shape": list(inputs["mask_shape"]),
        "brain_voxel_count": int(inputs["brain_voxel_count"]),
        "chosen_worker_count": min(
            context.workers, int(inputs["brain_voxel_count"])
        ),
        "matlab_version": context.matlab.version,
        "mexext": context.matlab.mexext,
        "model_name": _MODEL_NAME,
        "source_hashes": dict(sorted(source_hashes.items())),
        "base_artifacts": _base_preparation_artifact_hashes(context),
    }


def _validate_partial_state(
    context: NODDIContext,
    state: Mapping[str, object],
    inputs: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> None:
    expected = _partial_state_payload(context, inputs, source_hashes)
    if dict(state) != expected:
        raise NODDIError("stale or structurally inconsistent partial NODDI state")
    private = (
        context.stage_dir
        / "nifti_matlab"
        / "matlab"
        / "@file_array"
        / "private"
    )
    for stem in _MEX_STEMS:
        binary = private / f"{stem}.{context.matlab.mexext}"
        if os.path.lexists(binary):
            _require_regular_single_link(binary, "partial compiled NIfTI MEX")
            if binary.stat().st_size < 1:
                raise NODDIError("partial compiled NIfTI MEX is empty")
    roi = context.stage_dir / "NODDI_roi.mat"
    if os.path.lexists(roi):
        _validate_roi(
            roi,
            brain_voxels=int(inputs["brain_voxel_count"]),
            volumes=int(inputs["dwi_shape"][-1]),
            spatial_shape=tuple(inputs["dwi_shape"][:3]),
        )


def _base_preparation_artifact_hashes(context: NODDIContext) -> dict[str, str]:
    paths = (
        context.stage_dir / "eddy_dwi.nii",
        context.stage_dir / "cleaned_mask.nii",
        context.stage_dir / "bvals_rounded.txt",
        context.stage_dir / "eddy_rotated_bvecs.txt",
    )
    result = {
        path.relative_to(context.stage_dir).as_posix(): _sha256(path)
        for path in paths
    }
    result["nifti_matlab_tree_sha256"] = _tree_sha256(
        context.stage_dir / "nifti_matlab",
        allowed_mexext=context.matlab.mexext,
    )
    return result


def _preparation_runtime_outputs_complete(
    context: NODDIContext, inputs: Mapping[str, object]
) -> bool:
    private = (
        context.stage_dir
        / "nifti_matlab"
        / "matlab"
        / "@file_array"
        / "private"
    )
    binaries = tuple(
        private / f"{stem}.{context.matlab.mexext}" for stem in _MEX_STEMS
    )
    if not all(os.path.lexists(path) for path in binaries):
        return False
    _validate_compiled_mex(context)
    roi = context.stage_dir / "NODDI_roi.mat"
    if not os.path.lexists(roi):
        return False
    _validate_roi(
        roi,
        brain_voxels=int(inputs["brain_voxel_count"]),
        volumes=int(inputs["dwi_shape"][-1]),
        spatial_shape=tuple(inputs["dwi_shape"][:3]),
    )
    return True


def _load_image(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    _require_regular_single_link(path, label)
    try:
        return nib.load(path)
    except (OSError, ImageFileError, HeaderDataError, ImageDataError) as error:
        raise NODDIError(f"cannot read {label}: {path}") from error


def _stream_mask(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    try:
        values = np.asanyarray(image.dataobj)
    except (OSError, ValueError, TypeError, ImageDataError) as error:
        raise NODDIError("cannot read cleaned mask voxel data") from error
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise NODDIError("cleaned mask must contain finite numeric data")
    return np.asarray(values > 0, dtype=bool)


def _load_text(path: Path, label: str) -> np.ndarray:
    _require_regular_single_link(path, label)
    try:
        return np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    except (OSError, ValueError) as error:
        raise NODDIError(f"cannot read {label}: {path}") from error


def _validate_merge_map(
    path: Path, reference: nib.spatialimages.SpatialImage
) -> np.ndarray:
    image = _load_image(path, "NODDI output map")
    if tuple(image.shape) != tuple(reference.shape):
        raise NODDIError(f"NODDI map has wrong shape: {path.name}")
    if not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
        raise NODDIError(f"NODDI map affine mismatch: {path.name}")
    try:
        return np.asarray(image.dataobj, dtype=float)
    except (OSError, ValueError, TypeError, ImageDataError) as error:
        raise NODDIError(f"cannot read NODDI map: {path.name}") from error


def _validate_merge_destinations(context: NODDIContext) -> None:
    destinations = (
        *(context.stage_dir / name for name in _REQUIRED_MAPS),
        context.stage_dir / "NODDI_params.mat",
        context.stage_dir / "noddi_metrics.json",
        context.stage_dir / "merge_noddi.log",
    )
    for path in destinations:
        if os.path.lexists(path):
            _require_regular_single_link(path, "pre-existing NODDI merge output")


def _load_mat_arrays(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    try:
        from scipy.io import loadmat

        values = loadmat(path, variable_names=("error_code", "mlps"))
        errors = values.get("error_code")
        mlps = values.get("mlps")
        if errors is not None and mlps is not None:
            return (
                np.asarray(errors, dtype=float).reshape(-1),
                tuple(int(value) for value in np.asarray(mlps).shape),
            )
    except (NotImplementedError, OSError, ValueError):
        pass
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if "error_code" not in handle or "mlps" not in handle:
                raise NODDIError("NODDI parameter MAT lacks error_code or mlps")
            return (
                np.asarray(handle["error_code"], dtype=float).reshape(-1),
                tuple(int(value) for value in handle["mlps"].shape),
            )
    except OSError as error:
        raise NODDIError("NODDI parameter MAT is unreadable") from error


def _load_roi_indices(path: Path) -> np.ndarray:
    _require_regular_single_link(path, "NODDI ROI")
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if "idx" not in handle:
                raise NODDIError("NODDI ROI lacks idx")
            indices = np.asarray(handle["idx"], dtype=float)
    except OSError as error:
        raise NODDIError("NODDI ROI is unreadable") from error
    if indices.ndim != 2:
        raise NODDIError("NODDI ROI idx must be two-dimensional")
    if indices.shape[1] == 3:
        normalized = indices
    elif indices.shape[0] == 3:
        normalized = indices.T
    else:
        raise NODDIError("NODDI ROI idx must have three coordinate columns")
    if (
        not np.isfinite(normalized).all()
        or not np.array_equal(normalized, np.rint(normalized))
        or np.any(normalized < 1)
    ):
        raise NODDIError("NODDI ROI idx coordinates must be positive integers")
    return np.asarray(normalized, dtype=np.int64)


def _stop_workers(
    processes: Sequence[tuple[int, object, object]], grace_seconds: float
) -> None:
    running = []
    for _, process, _ in processes:
        try:
            if process.poll() is None:
                process.terminate()
                running.append(process)
        except Exception:
            running.append(process)
    deadline = time.monotonic() + float(grace_seconds)
    for process in running:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    for process in running:
        try:
            process.wait(timeout=max(0.1, float(grace_seconds)))
        except Exception:
            pass


def _close_worker_logs(
    processes: Sequence[tuple[int, object, object]]
) -> None:
    for _, _, handle in processes:
        try:
            handle.close()
        except Exception:
            pass


def _resume_instruction(context: NODDIContext) -> str:
    return (
        f"./run_pipeline.sh {shlex.quote(str(context.config.config_path))}"
    )


def _validated_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise NODDIError("MATLAB argv must be a non-empty string sequence")
    if any(not isinstance(item, str) for item in argv):
        raise NODDIError("MATLAB argv must contain only strings")
    return tuple(argv)


def _open_log(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise NODDIError("platform lacks safe no-follow log traversal support")
    absolute = Path(os.path.abspath(path))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    directory_fd = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            try:
                metadata = os.stat(
                    component, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                metadata = os.stat(
                    component, dir_fd=directory_fd, follow_symlinks=False
                )
            if stat.S_ISLNK(metadata.st_mode):
                raise NODDIError(f"MATLAB log parent is a symbolic link: {component}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise NODDIError(f"MATLAB log parent is not a directory: {component}")
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            existing = os.stat(
                absolute.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise NODDIError(f"MATLAB runtime log is unsafe: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(
                absolute.name, flags, 0o600, dir_fd=directory_fd
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise NODDIError(f"MATLAB runtime log is a symbolic link: {path}") from error
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise NODDIError(f"MATLAB runtime log is not a safe regular file: {path}")
        return descriptor
    finally:
        os.close(directory_fd)


def _ensure_safe_directory(path: Path, *, create: bool, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise NODDIError(f"{label} does not exist: {path}")
            try:
                current.mkdir()
            except FileExistsError:
                pass
            metadata = current.lstat()
        except OSError as error:
            raise NODDIError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise NODDIError(f"{label} contains a symbolic link: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise NODDIError(f"{label} contains a non-directory: {current}")


def _require_absent_safe(path: Path) -> None:
    _ensure_safe_directory(path.parent, create=False, label="output parent")
    if os.path.lexists(path):
        raise NODDIError(f"refusing to overwrite pre-existing output: {path}")


def _require_regular_single_link(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise NODDIError(f"{label} is missing or unreadable: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise NODDIError(f"{label} contains a symbolic link: {current}")
    if not stat.S_ISREG(metadata.st_mode):
        raise NODDIError(f"{label} must be a regular file: {path}")
    if metadata.st_nlink != 1:
        raise NODDIError(f"{label} must not be hard-linked: {path}")


def _safe_regular_file(path: Path) -> bool:
    try:
        _require_regular_single_link(path, "worker final")
    except NODDIError:
        return False
    return True


def _validate_worker_final(
    path: Path, worker: int, workers: int, total_rows: int
) -> bool:
    if not _safe_regular_file(path):
        return False
    block_size = math.ceil(total_rows / workers)
    start = (worker - 1) * block_size + 1
    end = min(worker * block_size, total_rows)
    local_rows = end - start + 1
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            required = {
                "gsps",
                "mlps",
                "fobj_gs",
                "fobj_ml",
                "error_code",
                "nextRow",
                "metadata",
            }
            if not required.issubset(handle.keys()):
                return False
            for name in ("gsps", "mlps"):
                if sorted(handle[name].shape) != sorted((local_rows, 8)):
                    return False
            for name in ("fobj_gs", "fobj_ml", "error_code"):
                shape = tuple(int(value) for value in handle[name].shape)
                if math.prod(shape) != local_rows:
                    return False
            if math.prod(handle["nextRow"].shape) != 1:
                return False
            next_row = float(np.asarray(handle["nextRow"]).reshape(-1)[0])
            if next_row != local_rows + 1:
                return False
            metadata = handle["metadata"]
            numeric_expected = {
                "workerIndexStored": worker,
                "numWorkersStored": workers,
                "globalStart": start,
                "globalEnd": end,
                "totalRows": total_rows,
                "numParams": 8,
            }
            if not hasattr(metadata, "keys"):
                return False
            for name, expected in numeric_expected.items():
                if name not in metadata:
                    return False
                values = np.asarray(metadata[name]).reshape(-1)
                if values.size != 1 or float(values[0]) != expected:
                    return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def _resolve(path: Path) -> Path:
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise NODDIError(f"cannot resolve path safely: {path}") from error


def _sha256(path: Path) -> str:
    _require_regular_single_link(path, "hashed file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise NODDIError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _tree_sha256(root: Path, *, allowed_mexext: str | None = None) -> str:
    _ensure_safe_directory(root, create=False, label="hashed source tree")
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        _require_regular_single_link(path, "hashed source-tree entry")
        relative = path.relative_to(root)
        if ".mex" in path.name.lower():
            allowed = {
                Path("matlab")
                / "@file_array"
                / "private"
                / f"{stem}.{allowed_mexext}"
                for stem in _MEX_STEMS
            } if allowed_mexext is not None else set()
            if relative not in allowed:
                raise NODDIError(f"source tree contains unexpected MEX binary: {path}")
            continue
        lower = path.name.lower()
        if lower.endswith((".class", ".o", ".obj", ".un~", ".pyc", ".log")):
            raise NODDIError(f"source tree contains forbidden generated file: {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_savetxt(path: Path, values: np.ndarray, fmt: str) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            np.savetxt(handle, values, fmt=fmt)
        os.replace(temporary, path)
    except (OSError, ValueError) as error:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise NODDIError(f"cannot write NODDI text output: {path}") from error


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
        os.replace(temporary, path)
    except (OSError, ValueError) as error:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise NODDIError(f"cannot write deterministic JSON: {path}") from error


def _read_json_regular(path: Path, label: str) -> dict[str, object]:
    _require_regular_single_link(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NODDIError(f"{label} is unreadable or malformed: {path}") from error
    if not isinstance(value, dict):
        raise NODDIError(f"{label} must contain a JSON object")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NODDIError(f"{label} must be a positive integer")
    return value
