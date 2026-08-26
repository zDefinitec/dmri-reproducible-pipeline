from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import h5py
from scipy.io import savemat

from dmri_pipeline.config import (
    AcquisitionConfig,
    AnalysisConfig,
    PipelineConfig,
    load_config,
)
from dmri_pipeline.noddi import (
    MATLABDiscoveryError,
    MATLABInstallation,
    NODDIContext,
    NODDIError,
    NODDIExternalCommandError,
    _default_command_runner,
    build_merge_command,
    build_prepare_command,
    build_worker_command,
    choose_noddi_workers,
    discover_matlab,
    launch_noddi_workers,
    merge_noddi,
    prepare_noddi,
)


def _config(tmp_path: Path, matlab: Path | None = None) -> PipelineConfig:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    return PipelineConfig(
        subject_id="SYN001",
        dwi_pa=inputs / "raw_pa.nii.gz",
        bvals=inputs / "raw.bval",
        bvecs=inputs / "raw.bvec",
        b0_ap=inputs / "raw_ap.nii.gz",
        output_root=tmp_path / "outputs",
        acquisition=AcquisitionConfig((0, -1, 0), (0, 1, 0), 0.07),
        analysis=AnalysisConfig(noddi_workers=2),
        fsldir=None,
        matlab_executable=matlab,
        config_path=tmp_path / "subject.yaml",
    )


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _installation(executable: Path) -> MATLABInstallation:
    return MATLABInstallation(
        executable=executable,
        version="24.1.0.2537033 (R2024a)",
        mexext="mexa64",
        optimization_toolbox=True,
        mex_configured=True,
    )


def _context(tmp_path: Path, *, workers: int = 2) -> NODDIContext:
    package_root = Path(__file__).parents[1]
    inputs = tmp_path / "upstream"
    inputs.mkdir(parents=True)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    dwi = np.ones((2, 2, 2, 8), dtype=np.float32)
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask.ravel()[:3] = 1
    eddy = inputs / "eddy.nii.gz"
    cleaned_mask = inputs / "mask.nii.gz"
    bvals = inputs / "bvals.txt"
    rotated = inputs / "eddy_rotated_bvecs.txt"
    nib.save(nib.Nifti1Image(dwi, affine), eddy)
    nib.save(nib.Nifti1Image(mask, affine), cleaned_mask)
    np.savetxt(bvals, [[0, 1000, 1000, 1000, 2000, 2000, 2000, 0]], fmt="%g")
    bvec_array = np.asarray(
        [
            [0, 1, 0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
            [0, 0, 1, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
            [0, 0, 0, 1, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        ]
    )
    np.savetxt(rotated, bvec_array, fmt="%.12g")
    matlab = _executable(tmp_path / "MATLAB.app" / "bin" / "matlab")
    return NODDIContext(
        config=_config(tmp_path),
        package_root=package_root,
        stage_dir=tmp_path / "08_noddi.work",
        eddy_dwi=eddy,
        cleaned_mask=cleaned_mask,
        bvals=bvals,
        rotated_bvecs=rotated,
        matlab=_installation(matlab),
        workers=workers,
    )


@pytest.mark.parametrize(
    ("cpu", "memory", "configured", "expected"),
    [
        (10, 16, "auto", 2),
        (12, 32, "auto", 4),
        (12, 64, "auto", 8),
        (12, 64, 3, 3),
    ],
)
def test_worker_selection_is_bounded(cpu, memory, configured, expected):
    assert choose_noddi_workers(cpu, memory, configured) == expected


@pytest.mark.parametrize(
    ("cpu", "memory", "configured"),
    [
        (True, 16, "auto"),
        (0, 16, "auto"),
        (8, float("nan"), "auto"),
        (8, 16, True),
        (8, 16, 0),
        (4, 16, 5),
    ],
)
def test_worker_selection_rejects_invalid_or_oversubscribed_values(
    cpu, memory, configured
):
    with pytest.raises(NODDIError):
        choose_noddi_workers(cpu, memory, configured)


def test_matlab_commands_are_generic_fresh_argv_and_escape_quotes(tmp_path):
    context = _context(tmp_path / "subject's run")
    object.__setattr__(context, "package_root", tmp_path / "portable-package")
    first = build_worker_command(context, worker=2, workers=3)
    second = build_worker_command(context, worker=2, workers=3)
    assert first == second and first is not second
    assert first[:2] == [str(context.matlab.executable), "-batch"]
    expression = first[2]
    assert "subject''s run" in expression
    assert "run_noddi_worker" in expression
    assert "2,3" in expression.replace(" ", "")
    combined = "\n".join(
        [
            *build_prepare_command(context),
            *first,
            *build_merge_command(context, 3),
        ]
    )
    assert str(context.package_root) in combined


def test_command_builders_reject_bad_worker_indices(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(NODDIError):
        build_worker_command(context, worker=0, workers=2)
    with pytest.raises(NODDIError):
        build_worker_command(context, worker=3, workers=2)
    with pytest.raises(NODDIError):
        build_merge_command(context, 0)


def test_prepare_command_compiles_only_missing_mex_and_creates_only_missing_roi(
    tmp_path,
):
    context = _context(tmp_path)
    private = (
        context.stage_dir
        / "nifti_matlab"
        / "matlab"
        / "@file_array"
        / "private"
    )
    private.mkdir(parents=True)
    for stem in ("file2mat", "mat2file", "init"):
        (private / f"{stem}.{context.matlab.mexext}").write_bytes(b"mex")
    _fake_prepare_outputs(context)
    (private / f"init.{context.matlab.mexext}").unlink()
    expression = build_prepare_command(context)[2]
    assert "init.c" in expression
    assert "file2mat.c" not in expression and "mat2file.c" not in expression
    assert "CreateROI(" not in expression

    (context.stage_dir / "NODDI_roi.mat").unlink()
    (private / f"mat2file.{context.matlab.mexext}").unlink()
    expression = build_prepare_command(context)[2]
    assert "init.c" in expression and "mat2file.c" in expression
    assert "file2mat.c" not in expression
    assert "CreateROI(" in expression


def test_discovery_uses_bounded_quick_and_full_mex_probes(
    tmp_path, monkeypatch
):
    app = tmp_path / "MATLAB_R2024a.app"
    executable = _executable(app / "bin" / "matlab")
    config = _config(tmp_path, app)
    before = os.environ.copy()
    seen: list[tuple[tuple[str, ...], float]] = []

    def fake_run(argv, **kwargs):
        seen.append((tuple(argv), kwargs["timeout"]))
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "__DMRI_MATLAB_VERSION__=24.1.0.2537033 (R2024a)\n"
                "__DMRI_MEXEXT__=mexa64\n"
                "__DMRI_OPT_INSTALLED__=1\n"
                "__DMRI_OPT_LICENSED__=1\n"
                "__DMRI_MEX_CONFIGURED__=1\n"
                "__DMRI_MEX_WORKS__=1\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("dmri_pipeline.noddi.subprocess.run", fake_run)
    found = discover_matlab(config)
    assert found.executable == executable
    assert found.mexext == "mexa64"
    assert [timeout for _, timeout in seen] == [30.0, 300.0]
    quick_expression = seen[0][0][2]
    mex_expression = seen[1][0][2]
    assert "__DMRI_MATLAB_VERSION__" in quick_expression
    assert "mex.getCompilerConfigurations" in quick_expression
    assert "dmri_mex_probe" not in quick_expression
    assert "mex('-silent','-outdir'" in mex_expression
    assert "addpath(d)" in mex_expression
    assert "dmri_mex_probe()" in mex_expression
    assert "__DMRI_MEX_WORKS__" in mex_expression
    assert os.environ == before


def test_discovery_invalid_explicit_path_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr("dmri_pipeline.noddi.shutil.which", lambda _: "/bin/matlab")
    with pytest.raises(MATLABDiscoveryError, match="explicit"):
        discover_matlab(_config(tmp_path, tmp_path / "missing.app"))


def test_discovery_uses_matlab_executable_environment_before_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = _executable(tmp_path / "MATLAB" / "R2025a" / "bin" / "matlab")
    monkeypatch.setenv("MATLAB_EXECUTABLE", str(executable))
    monkeypatch.setattr(
        "dmri_pipeline.noddi.shutil.which",
        lambda name: (_ for _ in ()).throw(AssertionError(name)),
    )
    monkeypatch.setattr(
        "dmri_pipeline.noddi.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "__DMRI_MATLAB_VERSION__=25.1\n"
                "__DMRI_MEXEXT__=mexa64\n"
                "__DMRI_OPT_INSTALLED__=1\n"
                "__DMRI_OPT_LICENSED__=1\n"
                "__DMRI_MEX_CONFIGURED__=1\n"
                "__DMRI_MEX_WORKS__=1\n"
            ),
            stderr="",
        ),
    )

    assert discover_matlab(_config(tmp_path)).executable == executable.resolve()


def test_invalid_matlab_environment_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MATLAB_EXECUTABLE", str(tmp_path / "missing-matlab"))
    monkeypatch.setattr("dmri_pipeline.noddi.shutil.which", lambda _: "/bin/matlab")

    with pytest.raises(MATLABDiscoveryError, match="MATLAB_EXECUTABLE"):
        discover_matlab(_config(tmp_path))


def test_explicit_matlab_rejects_expected_version_mismatch_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subject_config
) -> None:
    explicit = _executable(tmp_path / "explicit" / "bin" / "matlab")
    fallback = _executable(tmp_path / "fallback" / "bin" / "matlab")
    monkeypatch.setenv("MATLAB_EXECUTABLE", str(fallback))
    monkeypatch.setenv("DMRI_EXPECTED_MATLAB_VERSION", "25.1")
    seen: list[tuple[str, ...]] = []
    yaml_path = subject_config.config_path
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8")
        + "tools:\n"
        + f"  matlab_executable: {json.dumps(str(explicit))}\n",
        encoding="utf-8",
    )

    def fake_run(argv, **kwargs):
        seen.append(tuple(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "__DMRI_MATLAB_VERSION__=24.2\n"
                "__DMRI_MEXEXT__=mexa64\n"
                "__DMRI_OPT_INSTALLED__=1\n"
                "__DMRI_OPT_LICENSED__=1\n"
                "__DMRI_MEX_CONFIGURED__=1\n"
                "__DMRI_MEX_WORKS__=1\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("dmri_pipeline.noddi.subprocess.run", fake_run)

    with pytest.raises(MATLABDiscoveryError, match="version mismatch.*24.2"):
        discover_matlab(load_config(yaml_path))
    assert [argv[0] for argv in seen] == [str(explicit.resolve())]


@pytest.mark.parametrize(
    ("mexext", "mex_works", "match"),
    (
        ("mexmaca64", "1", "mexa64"),
        ("mexa64", "0", "compile.*load.*run"),
    ),
)
def test_discovery_rejects_incompatible_linux_mex_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mexext: str,
    mex_works: str,
    match: str,
) -> None:
    executable = _executable(tmp_path / "matlab")
    monkeypatch.setattr(
        "dmri_pipeline.noddi.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "__DMRI_MATLAB_VERSION__=25.1\n"
                f"__DMRI_MEXEXT__={mexext}\n"
                "__DMRI_OPT_INSTALLED__=1\n"
                "__DMRI_OPT_LICENSED__=1\n"
                "__DMRI_MEX_CONFIGURED__=1\n"
                f"__DMRI_MEX_WORKS__={mex_works}\n"
            ),
            stderr="",
        ),
    )

    with pytest.raises(MATLABDiscoveryError, match=match):
        discover_matlab(_config(tmp_path, executable))


@pytest.mark.parametrize(
    ("stdout", "match"),
    [
        (
            "__DMRI_MATLAB_VERSION__=R2024a\n"
            "__DMRI_MEXEXT__=mexa64\n"
            "__DMRI_OPT_INSTALLED__=0\n"
            "__DMRI_OPT_LICENSED__=0\n"
            "__DMRI_MEX_CONFIGURED__=1\n"
            "__DMRI_MEX_WORKS__=1\n",
            "Optimization Toolbox",
        ),
        (
            "__DMRI_MATLAB_VERSION__=R2024a\n"
            "__DMRI_MEXEXT__=mexa64\n"
            "__DMRI_OPT_INSTALLED__=1\n"
            "__DMRI_OPT_LICENSED__=1\n"
            "__DMRI_MEX_CONFIGURED__=0\n"
            "__DMRI_MEX_WORKS__=0\n",
            "MEX",
        ),
        ("ordinary MATLAB output\n", "sentinel"),
        (
            "__DMRI_MATLAB_VERSION__=R2024a\n"
            "__DMRI_MATLAB_VERSION__=R2024b\n"
            "__DMRI_MEXEXT__=mexa64\n"
            "__DMRI_OPT_INSTALLED__=1\n"
            "__DMRI_OPT_LICENSED__=1\n"
            "__DMRI_MEX_CONFIGURED__=1\n"
            "__DMRI_MEX_WORKS__=1\n",
            "sentinel",
        ),
    ],
)
def test_discovery_rejects_failed_capability_probes(
    tmp_path, monkeypatch, stdout, match
):
    executable = _executable(tmp_path / "matlab")
    monkeypatch.setattr(
        "dmri_pipeline.noddi.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=stdout, stderr=""
        ),
    )
    with pytest.raises(MATLABDiscoveryError, match=match):
        discover_matlab(_config(tmp_path, executable))


def test_discovery_normalizes_launch_timeout_and_nonzero(tmp_path, monkeypatch):
    executable = _executable(tmp_path / "matlab")
    config = _config(tmp_path, executable)
    monkeypatch.setattr(
        "dmri_pipeline.noddi.subprocess.run",
        lambda argv, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(argv, kwargs["timeout"])
        ),
    )
    with pytest.raises(MATLABDiscoveryError, match="timed out"):
        discover_matlab(config)

    monkeypatch.setattr(
        "dmri_pipeline.noddi.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 9, stdout="", stderr="license manager"
        ),
    )
    with pytest.raises(MATLABDiscoveryError, match="exit code 9"):
        discover_matlab(config)


def test_runtime_command_failure_has_typed_external_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dmri_pipeline.noddi.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 9),
    )

    with pytest.raises(NODDIExternalCommandError, match="exited with code 9"):
        _default_command_runner(
            (str(_executable(tmp_path / "matlab")), "-batch", "disp(1)"),
            tmp_path / "matlab.log",
        )


def _fake_prepare_outputs(context: NODDIContext) -> None:
    private = (
        context.stage_dir
        / "nifti_matlab"
        / "matlab"
        / "@file_array"
        / "private"
    )
    for stem in ("file2mat", "mat2file", "init"):
        (private / f"{stem}.{context.matlab.mexext}").write_bytes(b"synthetic mex")
    mask_values = np.asarray(nib.load(context.cleaned_mask).dataobj) > 0
    indices = np.argwhere(mask_values) + 1
    voxel_count = int(indices.shape[0])
    volumes = int(nib.load(context.eddy_dwi).shape[-1])
    with h5py.File(context.stage_dir / "NODDI_roi.mat", "w") as handle:
        handle.create_dataset("roi", data=np.ones((voxel_count, volumes)))
        handle.create_dataset("idx", data=indices)
        label_mask = np.zeros(mask_values.shape, dtype=np.uint32)
        for label, row in enumerate(indices, start=1):
            label_mask[tuple(row - 1)] = label
        handle.create_dataset("mask", data=label_mask)


def test_prepare_accepts_zero_or_unit_b0_vectors_and_rounds_half_away_from_zero(
    tmp_path,
):
    context = _context(tmp_path)
    bvals = np.loadtxt(context.bvals).reshape(-1)
    bvals[1] = 250
    np.savetxt(context.bvals, bvals[None, :], fmt="%g")
    bvecs = np.loadtxt(context.rotated_bvecs)
    bvecs[:, [0, 7]] = np.asarray([[1, 0, 0]]).T
    np.savetxt(context.rotated_bvecs, bvecs, fmt="%.12g")
    seen = []

    def runner(argv, log_path):
        seen.append(list(argv))
        _fake_prepare_outputs(context)
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    before = {
        path: path.read_bytes()
        for path in (
            context.eddy_dwi,
            context.cleaned_mask,
            context.bvals,
            context.rotated_bvecs,
        )
    }
    result = prepare_noddi(context)
    assert result.brain_voxel_count == 3
    rounded = np.loadtxt(context.stage_dir / "bvals_rounded.txt").reshape(-1)
    assert rounded[1] == 300
    assert np.loadtxt(context.stage_dir / "eddy_rotated_bvecs.txt").shape == (3, 8)
    assert all(path.read_bytes() == content for path, content in before.items())
    command = seen[0][2]
    assert all(name in command for name in ("file2mat.c", "mat2file.c", "init.c"))


def test_prepare_rejects_intermediate_b0_norm(tmp_path):
    context = _context(tmp_path)
    bvecs = np.loadtxt(context.rotated_bvecs)
    bvecs[:, 0] = [0.5, 0, 0]
    np.savetxt(context.rotated_bvecs, bvecs, fmt="%.12g")
    with pytest.raises(NODDIError, match="b0"):
        prepare_noddi(context)


def test_prepare_rejects_protocol_without_diffusion_weighting(tmp_path):
    context = _context(tmp_path)
    np.savetxt(context.bvals, np.zeros((1, 8)), fmt="%g")
    with pytest.raises(NODDIError, match="non-b0|diffusion"):
        prepare_noddi(context)


def test_prepare_rejects_nonfinite_roi_from_matlab(tmp_path):
    context = _context(tmp_path)

    def runner(argv, log_path):
        _fake_prepare_outputs(context)
        with h5py.File(context.stage_dir / "NODDI_roi.mat", "r+") as handle:
            handle["roi"][0, 0] = np.nan
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    with pytest.raises(NODDIError, match="ROI signal values must be finite"):
        prepare_noddi(context)


def test_prepare_resume_rejects_stale_input_hash_and_unsafe_destination(tmp_path):
    context = _context(tmp_path)

    def runner(argv, log_path):
        _fake_prepare_outputs(context)
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    prepare_noddi(context)
    context.bvals.write_text(
        context.bvals.read_text(encoding="utf-8") + " \n", encoding="utf-8"
    )
    with pytest.raises(NODDIError, match="stale"):
        prepare_noddi(context)

    other = _context(tmp_path / "unsafe")
    other.stage_dir.mkdir()
    target = tmp_path / "escaped"
    target.mkdir()
    (other.stage_dir / "nifti_matlab").symlink_to(target, target_is_directory=True)
    with pytest.raises(NODDIError, match="symbolic link|unsafe"):
        prepare_noddi(other)


def test_context_and_prepare_reject_hardlink_aliases_and_stage_symlink(tmp_path):
    context = _context(tmp_path)
    context.rotated_bvecs.unlink()
    os.link(context.bvals, context.rotated_bvecs)
    with pytest.raises(NODDIError, match="hard-linked"):
        replace(context)

    other = _context(tmp_path / "stage-link")
    escape = tmp_path / "escaped-stage"
    escape.mkdir()
    other.stage_dir.symlink_to(escape, target_is_directory=True)
    with pytest.raises(NODDIError, match="symbolic link"):
        prepare_noddi(other)


def test_resume_metadata_and_worker_state_reject_hardlinks_and_symlinks(tmp_path):
    context = _context(tmp_path)
    object.__setattr__(
        context,
        "command_runner",
        lambda argv, log_path: (
            _fake_prepare_outputs(context) or SimpleNamespace(returncode=0)
        ),
    )
    result = prepare_noddi(context)
    external = tmp_path / "metadata-copy.json"
    result.metadata.replace(external)
    os.link(external, result.metadata)
    with pytest.raises(NODDIError, match="hard-linked"):
        prepare_noddi(context)

    result.metadata.unlink()
    external.replace(result.metadata)
    checkpoint_target = tmp_path / "checkpoint-target.mat"
    checkpoint_target.write_bytes(b"checkpoint")
    (context.stage_dir / "worker_01_checkpoint.mat").symlink_to(checkpoint_target)
    with pytest.raises(NODDIError, match="symbolic link"):
        launch_noddi_workers(context)


def test_prepare_resumes_validated_partial_compilation_state(tmp_path):
    context = _context(tmp_path)
    attempts = 0

    def runner(argv, log_path):
        nonlocal attempts
        attempts += 1
        _fake_prepare_outputs(context)
        if attempts == 1:
            raise NODDIError("synthetic interruption after MATLAB outputs")
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    with pytest.raises(NODDIError, match="synthetic interruption"):
        prepare_noddi(context)
    result = prepare_noddi(context)
    assert not result.resumed
    assert attempts == 1
    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert "nifti_matlab_tree_sha256" in metadata["artifacts"]


class _FakeProcess:
    def __init__(self, returncodes: list[int | None], *, survive_terminate=False):
        self.returncodes = list(returncodes)
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.survive_terminate = survive_terminate

    def poll(self):
        if self.returncodes:
            value = self.returncodes.pop(0)
            if value is not None:
                self.returncode = value
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if self.terminated and not self.survive_terminate:
                self.returncode = -15
            elif self.killed:
                self.returncode = -9
            else:
                raise subprocess.TimeoutExpired("matlab", timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_launcher_caps_workers_to_roi_voxels_and_uses_no_shell(tmp_path):
    context = _context(tmp_path, workers=7)
    object.__setattr__(
        context,
        "command_runner",
        lambda argv, log_path: (
            _fake_prepare_outputs(context) or SimpleNamespace(returncode=0)
        ),
    )
    prepare_noddi(context)
    calls = []
    processes = [_FakeProcess([0]), _FakeProcess([0]), _FakeProcess([0])]

    def factory(argv, **kwargs):
        calls.append((list(argv), kwargs))
        worker = len(calls)
        with h5py.File(
            context.stage_dir / f"worker_{worker:02d}_final.mat", "w"
        ) as handle:
            local_rows = 1
            handle.create_dataset("gsps", data=np.ones((local_rows, 8)))
            handle.create_dataset("mlps", data=np.ones((local_rows, 8)))
            handle.create_dataset("fobj_gs", data=np.ones((local_rows, 1)))
            handle.create_dataset("fobj_ml", data=np.ones((local_rows, 1)))
            handle.create_dataset("error_code", data=np.zeros((local_rows, 1)))
            handle.create_dataset("nextRow", data=[[local_rows + 1]])
            metadata = handle.create_group("metadata")
            total = 3
            block = 1
            metadata.create_dataset("workerIndexStored", data=[[worker]])
            metadata.create_dataset("numWorkersStored", data=[[3]])
            metadata.create_dataset("globalStart", data=[[(worker - 1) * block + 1]])
            metadata.create_dataset("globalEnd", data=[[worker * block]])
            metadata.create_dataset("totalRows", data=[[total]])
            metadata.create_dataset("numParams", data=[[8]])
        return processes[len(calls) - 1]

    object.__setattr__(context, "process_factory", factory)
    result = launch_noddi_workers(context)
    assert result.worker_count == 3
    assert len(calls) == 3
    assert all(call[1]["shell"] is False for call in calls)
    assert all("stdout" in call[1] and call[1]["stderr"] is subprocess.STDOUT for call in calls)


def test_launcher_failure_terminates_then_kills_and_preserves_checkpoints(tmp_path):
    context = _context(tmp_path, workers=2)
    object.__setattr__(
        context,
        "command_runner",
        lambda argv, log_path: (
            _fake_prepare_outputs(context) or SimpleNamespace(returncode=0)
        ),
    )
    prepare_noddi(context)
    checkpoint = context.stage_dir / "worker_02_checkpoint.mat"
    checkpoint.write_bytes(b"checkpoint")
    failed = _FakeProcess([7])
    peer = _FakeProcess([None, None], survive_terminate=True)
    processes = [failed, peer]

    def factory(argv, **kwargs):
        return processes.pop(0)

    object.__setattr__(context, "process_factory", factory)
    object.__setattr__(context, "poll_interval_seconds", 0.0)
    object.__setattr__(context, "terminate_grace_seconds", 0.0)
    with pytest.raises(NODDIError) as caught:
        launch_noddi_workers(context)
    message = str(caught.value)
    assert "worker 1" in message
    assert "worker_01.log" in message
    assert "run_pipeline.sh" in message
    assert "--force-stage" not in message
    assert peer.terminated and peer.killed
    assert checkpoint.read_bytes() == b"checkpoint"


def _write_merge_outputs(context: NODDIContext) -> None:
    reference = nib.load(context.cleaned_mask)
    shape = reference.shape
    mask = np.asarray(reference.dataobj) > 0
    error = np.zeros(shape, dtype=np.float32)
    error[~mask] = 0
    maps = {
        "NODDI_odi.nii": np.where(mask, 0.2, 0).astype(np.float32),
        "NODDI_ficvf.nii": np.where(mask, 0.6, 0).astype(np.float32),
        "NODDI_fiso.nii": np.where(mask, 0.1, 0).astype(np.float32),
        "NODDI_fmin.nii": np.where(mask, 4.0, 0).astype(np.float32),
        "NODDI_error_code.nii": error,
        "NODDI_kappa.nii": np.where(mask, 1.0, 0).astype(np.float32),
        "NODDI_fibredirs_xvec.nii": np.where(mask, 1.0, 0).astype(np.float32),
        "NODDI_fibredirs_yvec.nii": np.zeros(shape, dtype=np.float32),
        "NODDI_fibredirs_zvec.nii": np.zeros(shape, dtype=np.float32),
    }
    for name, values in maps.items():
        nib.save(nib.Nifti1Image(values, reference.affine), context.stage_dir / name)
    savemat(
        context.stage_dir / "NODDI_params.mat",
        {"mlps": np.ones((3, 8)), "error_code": np.zeros((3, 1))},
    )
    (context.stage_dir / "noddi_metrics.json").write_text(
        json.dumps(
            {
                "total_voxels": 3,
                "success_count": 3,
                "error_999_count": 0,
                "other_error_count": 0,
                "worker_count": 2,
                "model_name": "WatsonSHStickTortIsoV_B0",
                "parameter_maps": sorted(maps),
                "objective_finite_count": 3,
                "objective_min": 4.0,
                "objective_max": 4.0,
                "objective_mean": 4.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _prepare_merge_fixture(context: NODDIContext) -> None:
    def runner(argv, log_path):
        _fake_prepare_outputs(context)
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    prepare_noddi(context)
    _write_merge_outputs(context)


def test_merge_runs_generic_command_and_validates_synthetic_outputs(tmp_path):
    context = _context(tmp_path, workers=2)
    _prepare_merge_fixture(context)
    seen = []

    def runner(argv, log_path):
        seen.append((list(argv), log_path))
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    result = merge_noddi(context)
    assert result.success_count == 3
    assert seen[0][0] == build_merge_command(context, 2)
    assert seen[0][1].name == "merge_noddi.log"


def test_merge_rejects_nonfinite_success_map_and_inconsistent_json(tmp_path):
    context = _context(tmp_path, workers=2)
    _prepare_merge_fixture(context)
    path = context.stage_dir / "NODDI_odi.nii"
    image = nib.load(path)
    values = np.asarray(image.dataobj).copy()
    values.ravel()[0] = np.nan
    nib.save(nib.Nifti1Image(values, image.affine), path)
    object.__setattr__(
        context,
        "command_runner",
        lambda argv, log_path: SimpleNamespace(returncode=0),
    )
    with pytest.raises(NODDIError, match="finite"):
        merge_noddi(context)


def test_merge_rejects_nonunit_success_fibre_direction(tmp_path):
    context = _context(tmp_path, workers=2)
    _prepare_merge_fixture(context)
    path = context.stage_dir / "NODDI_fibredirs_xvec.nii"
    image = nib.load(path)
    nib.save(
        nib.Nifti1Image(np.zeros(image.shape, dtype=np.float32), image.affine),
        path,
    )
    object.__setattr__(
        context,
        "command_runner",
        lambda argv, log_path: SimpleNamespace(returncode=0),
    )
    with pytest.raises(NODDIError, match="unit length"):
        merge_noddi(context)


def test_merge_rejects_upstream_mutation_during_matlab_command(tmp_path):
    context = _context(tmp_path, workers=2)
    _prepare_merge_fixture(context)

    def mutating_runner(argv, log_path):
        image = nib.load(context.eddy_dwi)
        values = np.asarray(image.dataobj).copy()
        values[0, 0, 0, 0] += 1
        nib.save(nib.Nifti1Image(values, image.affine, image.header), context.eddy_dwi)
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", mutating_runner)
    with pytest.raises(NODDIError, match="stale|changed|hash"):
        merge_noddi(context)


def test_merge_fails_closed_before_matlab_without_prepare_metadata(tmp_path):
    context = _context(tmp_path)
    context.stage_dir.mkdir()
    called = False

    def runner(argv, log_path):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    object.__setattr__(context, "command_runner", runner)
    with pytest.raises(NODDIError, match="preparation metadata"):
        merge_noddi(context)
    assert not called


def test_package_matlab_sources_and_vendor_are_generic_source_only():
    root = Path(__file__).parents[1]
    matlab_files = sorted((root / "scripts" / "matlab").glob("*.m"))
    assert {path.name for path in matlab_files} == {
        "CreateROI.m",
        "merge_noddi_workers.m",
        "run_noddi_worker.m",
    }
    source = "\n".join(path.read_text(encoding="utf-8") for path in matlab_files)
    assert "checkpointEvery = 500" in source
    assert "error_code(localRow) = 999" in source
    assert "blockSize = ceil(totalRows / numWorkers)" in source
    assert "expectedStart" in source and "expectedEnd" in source
    assert "-v7.3" in source
    worker_source = (root / "scripts" / "matlab" / "run_noddi_worker.m").read_text(
        encoding="utf-8"
    )
    assert "[mfilename('fullpath') '.m']" in worker_source
    assert "isfile(scriptPath)" in worker_source
    forbidden = []
    for path in (root / "vendor").rglob("*"):
        if path.is_file() and (
            ".mex" in path.suffix
            or path.suffix in {".class", ".pyc", ".un~", ".log"}
            or path.name in {".DS_Store"}
            or "__pycache__" in path.parts
        ):
            forbidden.append(path.relative_to(root).as_posix())
    assert forbidden == []
    expected = {
        "noddi_toolbox_v1.05": (
            64,
            "4185352f6c22274128b72d35dc23fe947907e61f57dc4b8eacb380046a9bd3ea",
        ),
        "nifti_matlab": (
            85,
            "eb5d50ff19e20bc51c1c9dcb03d8656682ef0d6f211a8ca64c6dd0896af0b9a5",
        ),
    }
    for name, (expected_count, expected_hash) in expected.items():
        vendor_root = root / "vendor" / name
        files = sorted(path for path in vendor_root.rglob("*") if path.is_file())
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(vendor_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        assert len(files) == expected_count
        assert digest.hexdigest() == expected_hash


def _real_matlab() -> Path:
    configured = os.environ.get("DMRI_TEST_MATLAB")
    if not configured:
        pytest.skip("set DMRI_TEST_MATLAB for real MATLAB scientific regressions")
    executable = Path(configured)
    if not executable.is_file():
        pytest.skip("DMRI_TEST_MATLAB is not an existing executable")
    return executable


def _run_matlab_json(expression: str, marker: str) -> dict:
    completed = subprocess.run(
        [str(_real_matlab()), "-batch", expression],
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    start = output.index(marker) + len(marker)
    end = output.index(f"{marker}_END", start)
    return json.loads(output[start:end])


def test_real_matlab_package_authored_erfi_fixed_vector_and_domain():
    watson = (
        Path(__file__).parents[1]
        / "vendor"
        / "noddi_toolbox_v1.05"
        / "models"
        / "watson"
    )
    quote = str(watson).replace("'", "''")
    marker = "NODDI_ERFI_JSON="
    payload = _run_matlab_json(
        f"addpath('{quote}','-begin');"
        "x=[0,sqrt(1e-5),sqrt(.1),sqrt(.5),1,sqrt(2),2,sqrt(8),"
        "5.7-eps(5.7),5.7,5.7+eps(5.7),sqrt(32),8,sqrt(128),NaN,Inf];"
        "y=NODDI_erfi(x);"
        "p=struct('values',y(1:14),'shape',size(y),"
        "'nonfinite_are_nan',all(isnan(y(15:16))));"
        f"fprintf('{marker}%s{marker}_END\\n',jsonencode(p));",
        marker,
    )
    expected = np.asarray(
        [
            0.0,
            0.0035682601265019961,
            0.36908447259575655,
            0.95343826925126063,
            1.6504257587975431,
            3.7731225115990203,
            18.564802414575595,
            643.12686953174261,
            13144107775352.664,
            12757862730865.061,
            12757862730865.148,
            8019126319102.8564,
            4.3972577040766024e26,
            1.9387140712791833e54,
        ]
    )
    actual = np.asarray(payload["values"], dtype=float)
    assert actual[0] == 0.0
    np.testing.assert_array_max_ulp(actual[1:], expected[1:], maxulp=8)
    assert payload["shape"] == [1, 16]
    assert payload["nonfinite_are_nan"] is True


def test_real_matlab_watson_callers_match_fixed_external_vectors():
    watson = (
        Path(__file__).parents[1]
        / "vendor"
        / "noddi_toolbox_v1.05"
        / "models"
        / "watson"
    )
    quote = str(watson).replace("'", "''")
    marker = "NODDI_CALLERS_JSON="
    actual = _run_matlab_json(
        f"addpath('{quote}','-begin');"
        "ks=[0;.5;8;30;30+eps(30);32.49-eps(32.49);32.49;64];"
        "[C,D]=WatsonSHCoeff(ks);"
        "kh=[0,1e-5,.5,8,30,32.49-eps(32.49),32.49,64];"
        "dw=zeros(2,numel(kh));J=zeros(2,3,numel(kh));"
        "for ii=1:numel(kh),"
        "[dw(:,ii),J(:,:,ii)]=WatsonHinderedDiffusionCoeff(1.7e-9,0.6e-9,kh(ii));"
        "end;"
        f"fprintf('{marker}%s{marker}_END\\n',"
        "jsonencode(struct('C',C,'D',D,'dw',dw,'J',J)));",
        marker,
    )
    expected_c = np.asarray(
        [
            [3.5449077018110318, 0, 0, 0, 0, 0, 0],
            [3.5449077018110318, 0.55167876838883156, 0.03563233886537507, 0.0015047890651607313, 4.7370460248080853e-5, 1.1911483782840216e-6, -3.7725857084377373e-8],
            [3.5449077018110318, 6.2866560594052245, 5.1053548609825068, 2.95194035328447, 1.3361423671732233, 0.49702249494948758, 0.15680642246104598],
            [3.5449077018110318, 7.5247888488608323, 8.937803627806904, 8.8811653011257849, 7.84362143291991, 6.3034625946023581, 4.6554181672640311],
            [3.5449077018110318, 7.52308, 8.93718, 8.87905, 7.84352, 6.30113, 4.65678],
            [3.5449077018110318, 7.5545684919972844, 9.0619012809299786, 9.1393795919868683, 8.2380453774098985, 6.7859006462672173, 5.1656963579198827],
            [3.5449077018110318, 7.5545684919972844, 9.0619012809299786, 9.13937959198687, 8.2380453774099, 6.78590064626722, 5.1656963579198862],
            [3.5449077018110318, 7.7392327893186739, 9.8204793951931979, 10.813154843541433, 10.97224705832469, 10.485350063469111, 9.54012998072277],
        ]
    )
    expected_d = np.asarray(
        [
            [0, 1.0568872793616029, 0, 0, 0, 0, 0],
            [0, 1.1454776817206771, 0.14587807735675498, 0.0091816457341436491, 0.00038397849193900672, -2.407686092379103, 1.0212871623748283e-6],
            [0, 0.23504368754327482, 0.535512206598931, 0.56293260999100392, 0.3844964678502017, 0.19446592368829527, 0.078587848596364432],
            [0, 0.012043770347861272, 0.053451893527480039, 0.10982565598875732, 0.1676713687414069, 0.20099110225171143, 0.21147555746077656],
            [0, 0.012043770347821862, 0.05345189352746435, 0.10982565598870303, 0.16767136874140617, 0.20099110225164468, 0.21147555746081337],
            [0, 0.16715468303029382, 0.10435997502398345, 0.2913499303978736, 0.164504387389235, 0.39792300012431558, 0.0888017772400279],
            [0, -0.17734724059084356, -0.024041566721814806, -0.13774988126866294, 0.13144217527119456, -0.0708393471190318, 0.33196461924351039],
            [0, -0.09143906973236525, -0.031503780981303742, -0.095200484722635717, 0.0049612859960006259, -0.059757867933970081, 0.07690607886614402],
        ]
    )
    expected_dw = np.asarray(
        [
            [9.6666666666666664e-10, 9.6666764450239429e-10, 1.0177051502150958e-9, 1.5482757664688153e-9, 1.6628214470828317e-9, 1.6507477593210963e-9, 1.6830717143736536e-9, 1.6914062499999999e-9],
            [9.6666666666666664e-10, 9.66666177750902e-10, 9.41147424892452e-10, 6.758621167655924e-10, 6.1858927645858413e-10, 6.2462612033945179e-10, 6.0846414281317325e-10, 6.0429687499999991e-10],
        ]
    )
    expected_j = np.asarray(
        [
            [
                [0.33333333333333331, 0.33333422226888842, 0.3797319547409963, 0.86206887860801407, 0.966201315529847, 0.95522523574645146, 0.98461064943059429, 0.9921875],
                [0.66666666666666663, 0.66666577772656055, 0.6202680452590037, 0.13793112139198593, 0.033798684470153023, 0.044774764253548496, 0.015389350569405751, 0.0078125],
                [9.777777777777778e-11, 9.7769168578041761e-11, 1.059737063732483e-10, 2.1745033644985835e-11, 1.1142277172558853e-12, 1.5464291955262454e-11, -1.6407254982261005e-11, -8.459472656249999e-12],
            ],
            [
                [0.33333333333333331, 0.33333288886328027, 0.31013402262950185, 0.068965560695992967, 0.016899342235076512, 0.022387382126774248, 0.0076946752847028756, 0.00390625],
                [0.66666666666666663, 0.66666711113494426, 0.68986597737049815, 0.93103443930400709, 0.9831006577649235, 0.97761261787322584, 0.9923053247152972, 0.99609375],
                [-4.888888888888889e-11, -4.8884794206780831e-11, -5.2986853186624149e-11, -1.087251682249293e-11, -5.5711385862794266e-13, -7.7321459776312268e-12, 8.2036274911305e-12, 4.2297363281250011e-12],
            ],
        ]
    )
    np.testing.assert_allclose(actual["C"], expected_c, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual["D"], expected_d, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual["dw"], expected_dw, rtol=1e-12, atol=1e-22)
    np.testing.assert_allclose(actual["J"], expected_j, rtol=1e-12, atol=1e-22)


def test_real_matlab_worker_smoke_hashes_actual_m_file_and_uses_stub_fit(tmp_path):
    executable = _real_matlab()

    source_worker = (
        Path(__file__).parents[1]
        / "scripts"
        / "matlab"
        / "run_noddi_worker.m"
    )
    fake_package = tmp_path / "portable-package"
    script_dir = fake_package / "scripts" / "matlab"
    fitting = fake_package / "vendor" / "noddi_toolbox_v1.05" / "fitting"
    models = fake_package / "vendor" / "noddi_toolbox_v1.05" / "models"
    script_dir.mkdir(parents=True)
    fitting.mkdir(parents=True)
    models.mkdir(parents=True)
    shutil.copy2(source_worker, script_dir / source_worker.name)
    (fitting / "FSL2Protocol.m").write_text(
        "function protocol=FSL2Protocol(varargin)\nprotocol=struct();\nend\n",
        encoding="utf-8",
    )
    (fitting / "ThreeStageFittingVoxel.m").write_text(
        "function [g,fg,m,fm,e]=ThreeStageFittingVoxel(varargin)\n"
        "g=1:8; fg=1; m=1:8; fm=2; e=0;\nend\n",
        encoding="utf-8",
    )
    (models / "MakeModel.m").write_text(
        "function model=MakeModel(name)\n"
        "model=struct('name',name,'numParams',8);\nend\n",
        encoding="utf-8",
    )

    stage = tmp_path / "stage"
    private = stage / "nifti_matlab" / "matlab" / "@file_array" / "private"
    private.mkdir(parents=True)
    np.savetxt(stage / "bvals_rounded.txt", [[0]], fmt="%d")
    np.savetxt(stage / "eddy_rotated_bvecs.txt", [[0], [0], [0]], fmt="%d")
    (stage / "noddi_prepare.json").write_text("{}\n", encoding="utf-8")

    quote = lambda path: str(path).replace("'", "''")
    expression = (
        f"root='{quote(stage)}';"
        "roi=1;mask=uint32(1);idx=[1 1 1];"
        "save(fullfile(root,'NODDI_roi.mat'),'roi','mask','idx','-v7.3');"
        "names={'file2mat','mat2file','init'};"
        "for i=1:numel(names),"
        "p=fullfile(root,'nifti_matlab','matlab','@file_array','private',"
        "[names{i} '.' mexext]);f=fopen(p,'w');fwrite(f,uint8(1));fclose(f);end;"
        f"addpath('{quote(script_dir)}','-begin');"
        "run_noddi_worker(root,1,1)"
    )
    completed = subprocess.run(
        [str(executable), "-batch", expression],
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (stage / "worker_01_final.mat").is_file()
