from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
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


def _run_software_loader(
    tmp_path: Path,
    config_text: str,
    *,
    inherited: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "private-software.sh"
    config.write_text(config_text, encoding="utf-8")
    fake_bin = tmp_path / "loader-bin"
    fake_bin.mkdir()
    _write_linux_uname_and_gnu_stat(fake_bin)
    environment = os.environ.copy()
    environment.update(inherited or {})
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["DMRI_SOFTWARE_CONFIG"] = str(config)
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                "set -euo pipefail; "
                "fail() { printf 'FAIL: %s\\n' \"$*\" >&2; exit 30; }; "
                f"source {PACKAGE_ROOT / 'scripts' / 'rocky_environment.sh'}; "
                "load_software_config"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_platform_helper(
    tmp_path: Path, release: Path, uname: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                "set -euo pipefail; "
                "fail() { printf 'FAIL: %s\\n' \"$*\" >&2; exit 30; }; "
                f"source {PACKAGE_ROOT / 'scripts' / 'rocky_environment.sh'}; "
                "check_rocky_platform \"$1\" \"$2\""
            ),
            "platform-helper",
            str(release),
            str(uname),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def test_platform_helper_accepts_explicit_rocky_fixture(tmp_path: Path) -> None:
    fake_bin = tmp_path / "platform-bin"
    fake_bin.mkdir()
    _write_linux_uname_and_gnu_stat(fake_bin)

    result = _run_platform_helper(
        tmp_path, _write_rocky_release(tmp_path), fake_bin / "uname"
    )

    assert result.returncode == 0, result.stderr
    assert "Rocky Linux 9.7 x86_64" in result.stdout


@pytest.mark.parametrize("wrapper", ("setup_rocky.sh", "run_pipeline.sh"))
@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin entry-point regression")
def test_public_wrapper_cannot_use_inherited_platform_bypass(
    tmp_path: Path, wrapper: str
) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    uname_capture = tmp_path / "path-uname-was-called"
    environment["UNAME_CAPTURE"] = str(uname_capture)
    fake_uname = Path(environment["PATH"].split(":", 1)[0]) / "uname"
    fake_uname.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'called\\n' > \"$UNAME_CAPTURE\"\n"
        "case \"${1:-}\" in\n"
        "  -s) printf '%s\\n' Linux ;;\n"
        "  -m) printf '%s\\n' x86_64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_uname.chmod(0o755)
    arguments = ["--check"] if wrapper == "setup_rocky.sh" else ["subject.yaml"]

    result = subprocess.run(
        [str(PACKAGE_ROOT / wrapper), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 30
    assert "/etc/os-release" in result.stderr
    assert environment["DMRI_OS_RELEASE_FILE"] not in result.stderr
    assert not uname_capture.exists()


def _run_internal_wrapper(
    wrapper: str,
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    test_environment = environment.copy()
    test_environment["DMRI_TEST_WRAPPER"] = str(PACKAGE_ROOT / wrapper)
    test_environment["DMRI_TEST_RELEASE"] = environment[
        "DMRI_OS_RELEASE_FILE"
    ]
    test_environment["DMRI_TEST_UNAME"] = str(
        Path(environment["PATH"].split(":", 1)[0]) / "uname"
    )
    function = (
        "_dmri_setup_rocky_main"
        if wrapper == "setup_rocky.sh"
        else "_dmri_run_pipeline_main"
    )
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "$DMRI_TEST_WRAPPER"; {function} '
                '"$DMRI_TEST_RELEASE" "$DMRI_TEST_UNAME" "$@"'
            ),
            "internal-wrapper",
            *arguments,
        ],
        cwd=cwd,
        env=test_environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "omitted",
    (
        "CONDA_EXE",
        "FSLDIR",
        "MATLAB_EXECUTABLE",
        "DMRI_EXPECTED_FSL_VERSION",
        "DMRI_EXPECTED_MATLAB_VERSION",
    ),
)
def test_software_config_omission_cannot_inherit_parent_value(
    tmp_path: Path, omitted: str
) -> None:
    values = {
        "CONDA_EXE": "/configured/conda",
        "FSLDIR": "/configured/fsl",
        "MATLAB_EXECUTABLE": "/configured/matlab",
        "DMRI_EXPECTED_FSL_VERSION": "6.0.7.18",
        "DMRI_EXPECTED_MATLAB_VERSION": "25.1",
    }
    config_text = "".join(
        f"export {name}={value!r}\n"
        for name, value in values.items()
        if name != omitted
    )
    inherited = {name: f"stale-{name}" for name in values}

    result = _run_software_loader(
        tmp_path, config_text, inherited=inherited
    )

    assert result.returncode == 30
    assert f"missing {omitted}" in result.stderr


def test_software_config_syntax_error_is_configuration_error(tmp_path: Path) -> None:
    result = _run_software_loader(
        tmp_path,
        "export CONDA_EXE=/configured/conda\nif then\n",
    )

    assert result.returncode == 30
    assert "could not be loaded" in result.stderr


def test_software_config_execution_error_is_configuration_error(
    tmp_path: Path,
) -> None:
    result = _run_software_loader(
        tmp_path,
        "false\nexport CONDA_EXE=/configured/conda\n",
    )

    assert result.returncode == 30
    assert "could not be loaded" in result.stderr


def test_software_config_exit_is_configuration_error(tmp_path: Path) -> None:
    result = _run_software_loader(
        tmp_path,
        "printf 'config-noise\\n'\nexit 7\n",
    )

    assert result.returncode == 30
    assert result.stdout == ""
    assert "could not be loaded" in result.stderr


def test_software_config_stdout_and_metacharacters_do_not_corrupt_import(
    tmp_path: Path,
) -> None:
    result = _run_software_loader(
        tmp_path,
        "printf 'config-noise\\n'\n"
        "export CONDA_EXE='/configured/conda with spaces'\n"
        "export FSLDIR='/configured/fsl:one'\n"
        "export MATLAB_EXECUTABLE='/configured/matlab'\n"
        "export DMRI_EXPECTED_FSL_VERSION='6.0.7.18'\n"
        "export DMRI_EXPECTED_MATLAB_VERSION='25.1; exit 9'\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert "config-noise" in result.stderr


def test_software_config_runs_without_unrelated_parent_environment(
    tmp_path: Path,
) -> None:
    result = _run_software_loader(
        tmp_path,
        "export CONDA_EXE=\"$PARENT_ONLY\"\n"
        "export FSLDIR=/configured/fsl\n"
        "export MATLAB_EXECUTABLE=/configured/matlab\n"
        "export DMRI_EXPECTED_FSL_VERSION=6.0.7.18\n"
        "export DMRI_EXPECTED_MATLAB_VERSION=25.1\n",
        inherited={"PARENT_ONLY": "/stale/conda"},
    )

    assert result.returncode == 30
    assert "could not be loaded" in result.stderr


@pytest.mark.parametrize("wrapper", ("setup_rocky.sh", "run_pipeline.sh"))
def test_wrapper_normalizes_software_config_exit(
    tmp_path: Path, wrapper: str
) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    Path(environment["DMRI_SOFTWARE_CONFIG"]).write_text(
        "printf 'config-noise\\n'\nexit 7\n", encoding="utf-8"
    )
    arguments = ["--check"] if wrapper == "setup_rocky.sh" else ["subject.yaml"]

    result = _run_internal_wrapper(
        wrapper, arguments, cwd=tmp_path, environment=environment
    )

    assert result.returncode == 30
    assert "config-noise" not in result.stdout
    assert "could not be loaded" in result.stderr


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

    result = _run_internal_wrapper(
        "run_pipeline.sh",
        ["--dry-run", config.name],
        cwd=tmp_path,
        environment=environment,
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

    result = _run_internal_wrapper(
        "run_pipeline.sh",
        ["relative.yaml"],
        cwd=tmp_path,
        environment=environment,
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

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
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

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
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

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
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

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode != 0
    assert "compile" in result.stderr.lower() or "MEX" in result.stderr


def test_setup_rejects_rocky_id_and_version_with_a_non_rocky_name(
    tmp_path: Path,
) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    release = tmp_path / "os-release"
    release.write_text(
        'NAME="Rocky-like Linux"\nID="rocky"\nVERSION_ID="9.7"\n',
        encoding="utf-8",
    )
    environment["DMRI_OS_RELEASE_FILE"] = str(release)

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode == 30
    assert "Rocky Linux 9.7" in result.stderr


def test_setup_rejects_mexext_with_mexa64_prefix(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    matlab = Path(environment["PATH"].split(":", 1)[0]) / "matlab"
    matlab.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "'__DMRI_MATLAB_VERSION__25.1' "
        "'__DMRI_MEXEXT__mexa64-invalid' "
        "'__DMRI_OPT_INSTALLED__1' "
        "'__DMRI_OPT_LICENSED__1' "
        "'__DMRI_MEX_CONFIGURED__1' "
        "'__DMRI_MEX_WORKS__1'\n",
        encoding="utf-8",
    )
    matlab.chmod(0o755)

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode == 30
    assert "mexa64" in result.stderr


def test_run_wrapper_requires_absolute_readable_software_config(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    environment.pop("DMRI_SOFTWARE_CONFIG", None)

    result = _run_internal_wrapper(
        "run_pipeline.sh",
        ["subject.yaml"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode == 30
    assert "DMRI_SOFTWARE_CONFIG" in result.stderr


def test_setup_rejects_group_or_world_writable_software_config(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    Path(environment["DMRI_SOFTWARE_CONFIG"]).chmod(0o666)

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode == 30
    assert "group- or world-writable" in result.stderr


def test_setup_rejects_non_rocky_release(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    release = tmp_path / "os-release"
    release.write_text('ID="ubuntu"\nVERSION_ID="24.04"\n', encoding="utf-8")
    environment["DMRI_OS_RELEASE_FILE"] = str(release)

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode != 0
    assert "Rocky Linux 9.7" in result.stderr


def test_run_wrapper_rejects_non_rocky_release(tmp_path: Path) -> None:
    environment, _ = _setup_check_environment(tmp_path)
    release = tmp_path / "os-release"
    release.write_text('ID="ubuntu"\nVERSION_ID="24.04"\n', encoding="utf-8")
    environment["DMRI_OS_RELEASE_FILE"] = str(release)

    result = _run_internal_wrapper(
        "run_pipeline.sh",
        ["subject.yaml"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode == 30
    assert "Rocky Linux 9.7" in result.stderr


def test_setup_rejects_external_version_mismatch(tmp_path: Path) -> None:
    environment, fsldir = _setup_check_environment(tmp_path)
    (fsldir / "etc" / "fslversion").write_text("6.0.7.17\n", encoding="utf-8")

    result = _run_internal_wrapper(
        "setup_rocky.sh",
        ["--check"],
        cwd=tmp_path,
        environment=environment,
    )

    assert result.returncode != 0
    assert "FSL version mismatch" in result.stderr
