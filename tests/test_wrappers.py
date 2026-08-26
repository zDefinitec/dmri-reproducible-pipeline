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


def _write_software_config(
    tmp_path: Path,
    *,
    conda: Path,
    fsldir: Path,
    matlab: Path,
) -> Path:
    config = tmp_path / "dmri-rocky9.sh"
    config.write_text(
        "\n".join(
            (
                f'export CONDA_EXE="{conda}"',
                f'export FSLDIR="{fsldir}"',
                f'export MATLAB_EXECUTABLE="{matlab}"',
                'export DMRI_EXPECTED_FSL_VERSION="6.0.7.18"',
                'export DMRI_EXPECTED_MATLAB_VERSION="25.1"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return config


def _write_rocky_release(tmp_path: Path) -> Path:
    release = tmp_path / "os-release"
    release.write_text(
        'NAME="Rocky Linux"\nID="rocky"\nVERSION_ID="9.7"\n',
        encoding="utf-8",
    )
    return release


def _write_linux_uname_and_gnu_stat(fake_bin: Path) -> None:
    uname = fake_bin / "uname"
    uname.write_text(
        "#!/usr/bin/env bash\n"
        "case \"${1:-}\" in\n"
        "  -s) printf '%s\\n' Linux ;;\n"
        "  -m) printf '%s\\n' x86_64 ;;\n"
        "  *) exec /usr/bin/uname \"$@\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    uname.chmod(0o755)
    stat = fake_bin / "stat"
    stat.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == '-c' ]]; then\n"
        "  case \"${2:-}\" in\n"
        "    '%u') exec /usr/bin/stat -f '%u' \"$3\" ;;\n"
        "    '%a') exec /usr/bin/stat -f '%OLp' \"$3\" ;;\n"
        "  esac\n"
        "fi\n"
        "exec /usr/bin/stat \"$@\"\n",
        encoding="utf-8",
    )
    stat.chmod(0o755)


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
    fsldir = tmp_path / "fsl"
    matlab = fake_bin / "matlab"
    matlab.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    matlab.chmod(0o755)
    config = tmp_path / "a config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    _write_linux_uname_and_gnu_stat(fake_bin)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["CAPTURE"] = str(capture)
    environment["DMRI_SOFTWARE_CONFIG"] = str(
        _write_software_config(
            tmp_path, conda=conda, fsldir=fsldir, matlab=matlab
        )
    )
    environment["DMRI_OS_RELEASE_FILE"] = str(_write_rocky_release(tmp_path))

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
    fsldir = tmp_path / "fsl"
    matlab = fake_bin / "matlab"
    matlab.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    matlab.chmod(0o755)
    environment = os.environ.copy()
    _write_linux_uname_and_gnu_stat(fake_bin)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["DMRI_SOFTWARE_CONFIG"] = str(
        _write_software_config(
            tmp_path, conda=conda, fsldir=fsldir, matlab=matlab
        )
    )
    environment["DMRI_OS_RELEASE_FILE"] = str(_write_rocky_release(tmp_path))

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
        "'__DMRI_MEXEXT__mexa64' "
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
    (fsldir / "etc" / "fslversion").write_text("6.0.7.18\n", encoding="utf-8")
    environment = os.environ.copy()
    _write_linux_uname_and_gnu_stat(fake_bin)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["MATLAB_ARGS_CAPTURE"] = str(tmp_path / "matlab-args.txt")
    environment["DMRI_SOFTWARE_CONFIG"] = str(
        _write_software_config(
            tmp_path, conda=conda, fsldir=fsldir, matlab=matlab
        )
    )
    environment["DMRI_OS_RELEASE_FILE"] = str(_write_rocky_release(tmp_path))
    return environment, fsldir


def test_setup_check_succeeds_with_complete_safe_fake_tools(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
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
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "applywarp" in result.stderr


def test_setup_check_uses_matlab_executable_from_software_config(
    tmp_path: Path,
) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    matlab = Path(environment["PATH"].split(":", 1)[0]) / "matlab"
    explicit = matlab.with_name("custom-matlab")
    matlab.rename(explicit)
    environment["DMRI_SOFTWARE_CONFIG"] = str(
        _write_software_config(
            tmp_path,
            conda=Path(environment["PATH"].split(":", 1)[0]) / "conda",
            fsldir=Path(environment["DMRI_SOFTWARE_CONFIG"]).parent / "fsl",
            matlab=explicit,
        )
    )

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
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
        "'__DMRI_MEXEXT__mexa64' "
        "'__DMRI_OPT_INSTALLED__1' "
        "'__DMRI_OPT_LICENSED__1' "
        "'__DMRI_MEX_CONFIGURED__1' "
        "'__DMRI_MEX_WORKS__0'\n",
        encoding="utf-8",
    )
    matlab.chmod(0o755)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "compile" in result.stderr.lower() or "MEX" in result.stderr


def test_run_wrapper_requires_absolute_readable_software_config(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    environment.pop("DMRI_SOFTWARE_CONFIG", None)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "run_pipeline.sh"), "subject.yaml"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 30
    assert "DMRI_SOFTWARE_CONFIG" in result.stderr


def test_setup_rejects_group_or_world_writable_software_config(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    Path(environment["DMRI_SOFTWARE_CONFIG"]).chmod(0o666)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 30
    assert "group- or world-writable" in result.stderr


def test_setup_rejects_non_rocky_release(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    release = tmp_path / "os-release"
    release.write_text('ID="ubuntu"\nVERSION_ID="24.04"\n', encoding="utf-8")
    environment["DMRI_OS_RELEASE_FILE"] = str(release)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Rocky Linux 9.7" in result.stderr


def test_run_wrapper_rejects_non_rocky_release(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    release = tmp_path / "os-release"
    release.write_text('ID="ubuntu"\nVERSION_ID="24.04"\n', encoding="utf-8")
    environment["DMRI_OS_RELEASE_FILE"] = str(release)

    result = subprocess.run(
        [str(PACKAGE_ROOT / "run_pipeline.sh"), "subject.yaml"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 30
    assert "Rocky Linux 9.7" in result.stderr


def test_setup_rejects_external_version_mismatch(tmp_path: Path) -> None:
    environment, fsldir = _setup_check_environment(tmp_path)
    (fsldir / "etc" / "fslversion").write_text("6.0.7.17\n", encoding="utf-8")

    result = subprocess.run(
        [str(PACKAGE_ROOT / "setup_rocky.sh"), "--check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "FSL version mismatch" in result.stderr
