from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
FSL_COMMANDS = (
    "topup",
    "applytopup",
    "bet",
    "fslmaths",
    "eddy_quad",
    "flirt",
    "fnirt",
    "invwarp",
    "applywarp",
    "eddy",
)


def test_environment_and_example_paths_use_public_one_command_contract() -> None:
    environment = yaml.safe_load(
        (PACKAGE_ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    example = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "subject.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert environment["name"] == "dmri-repro"
    assert example["output_root"] == "../outputs"


def test_run_wrapper_is_relocation_safe_preserves_argv_exit_and_bytecode_setting(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture.txt"
    conda = fake_bin / "conda"
    conda.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == \"run -n dmri-repro python -c import sys\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$PYTHONDONTWRITEBYTECODE\" \"$PWD\" \"$@\" > \"$CAPTURE\"\n"
        "exit 7\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)
    config = tmp_path / "a config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("CONDA_EXE", None)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["CAPTURE"] = str(capture)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "run_pipeline.sh"), "--dry-run", config.name],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "1",
        str(tmp_path),
        "run",
        "-n",
        "dmri-repro",
        "python",
        "-m",
        "dmri_pipeline.cli",
        "--dry-run",
        config.name,
    ]


def test_run_wrapper_returns_dependency_code_when_environment_is_missing(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'EnvironmentNameNotFound: dmri-repro' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("CONDA_EXE", None)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = subprocess.run(
        [str(PACKAGE_ROOT / "run_pipeline.sh"), "relative.yaml"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 30
    assert "dmri-repro" in result.stderr


def _setup_check_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'OK: pinned Python imports and versions'\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)
    matlab = fake_bin / "matlab"
    matlab.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ -n \"${MATLAB_ARGS_CAPTURE:-}\" ]]; then\n"
        "  printf '%s\\n' \"$@\" > \"$MATLAB_ARGS_CAPTURE\"\n"
        "fi\n"
        "printf '%s\\n' "
        "'__DMRI_MATLAB_VERSION__25.1' "
        "'__DMRI_MEXEXT__mexmaca64' "
        "'__DMRI_OPT_INSTALLED__1' "
        "'__DMRI_OPT_LICENSED__1' "
        "'__DMRI_MEX_CONFIGURED__1' "
        "'__DMRI_MEX_WORKS__1'\n",
        encoding="utf-8",
    )
    matlab.chmod(0o755)
    fsldir = tmp_path / "fsl"
    (fsldir / "bin").mkdir(parents=True)
    for name in FSL_COMMANDS:
        executable = fsldir / "bin" / name
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    for relative in (
        "etc/flirtsch/b02b0.cnf",
        "etc/flirtsch/b02b0_1.cnf",
        "etc/flirtsch/FA_2_FMRIB58_1mm.cnf",
        "data/standard/FMRIB58_FA_1mm.nii.gz",
    ):
        path = fsldir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("CONDA_EXE", None)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FSLDIR"] = str(fsldir)
    environment["MATLAB_ARGS_CAPTURE"] = str(tmp_path / "matlab-args.txt")
    return environment, fsldir


def test_setup_check_succeeds_with_complete_safe_fake_tools(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_macos.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "packaged JHU image/XML hashes" in result.stdout
    assert "available disk" in result.stdout
    matlab_argv = (tmp_path / "matlab-args.txt").read_text(encoding="utf-8")
    assert "tempname" in matlab_argv
    assert "mex('-silent','-outdir'" in matlab_argv
    assert "dmri_mex_probe()" in matlab_argv


def test_setup_check_names_a_missing_fsl_component(tmp_path: Path) -> None:
    environment, fsldir = _setup_check_environment(tmp_path)
    (fsldir / "bin" / "applywarp").unlink()

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_macos.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "applywarp" in result.stderr


def test_setup_check_honors_explicit_matlab_executable_override(
    tmp_path: Path,
) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    matlab = Path(environment["PATH"].split(":", 1)[0]) / "matlab"
    explicit = matlab.with_name("custom-matlab")
    matlab.rename(explicit)
    environment["MATLAB_EXECUTABLE"] = str(explicit)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_macos.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "MATLAB" in result.stdout


def test_setup_check_fails_when_mex_cannot_compile_and_run(
    tmp_path: Path,
) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    matlab = Path(environment["PATH"].split(":", 1)[0]) / "matlab"
    matlab.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "'__DMRI_MATLAB_VERSION__25.1' "
        "'__DMRI_MEXEXT__mexmaca64' "
        "'__DMRI_OPT_INSTALLED__1' "
        "'__DMRI_OPT_LICENSED__1' "
        "'__DMRI_MEX_CONFIGURED__1' "
        "'__DMRI_MEX_WORKS__0'\n",
        encoding="utf-8",
    )
    matlab.chmod(0o755)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_macos.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "compile" in result.stderr.lower() or "MEX" in result.stderr
