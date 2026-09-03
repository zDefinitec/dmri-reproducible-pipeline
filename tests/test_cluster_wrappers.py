from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _shell_assignment(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}\n"


def _write_executable(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def cluster_package(tmp_path: Path) -> dict[str, Path | dict[str, str]]:
    """Relocate the real wrappers while faking only their external boundaries."""
    package = tmp_path / "relocated package"
    cluster_source = REPOSITORY_ROOT / "scripts" / "cluster"
    assert cluster_source.is_dir(), "cluster wrappers have not been implemented"
    shutil.copytree(cluster_source, package / "scripts" / "cluster")
    shutil.copy2(
        REPOSITORY_ROOT / "scripts" / "rocky_environment.sh",
        package / "scripts" / "rocky_environment.sh",
    )

    fake_pipeline = _write_executable(
        package / "run_pipeline.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ ${1-} == --print-cluster-context ]]; then\n"
        "  printf '%s\\0' \"$@\" >> \"${CONTEXT_CAPTURE}\"\n"
        "  printf '%s\\0' \"${DMRI_SOFTWARE_CONFIG-}\" >> \"${CONTEXT_SOFTWARE_CAPTURE}\"\n"
        "  printf '%s\\n' \"${FAKE_CONTEXT_JSON}\"\n"
        "  exit \"${FAKE_CONTEXT_STATUS:-0}\"\n"
        "fi\n"
        "printf '%s\\0' \"${DMRI_SOFTWARE_CONFIG-}\" >> \"${PIPELINE_SOFTWARE_CAPTURE}\"\n"
        "printf '%s\\0' \"$@\" >> \"${PIPELINE_CAPTURE}\"\n"
        "exit \"${FAKE_PIPELINE_STATUS:-0}\"\n",
    )
    assert fake_pipeline.is_file()

    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "stat",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ ${1-} == -c && ${2-} == '%u' "
        "&& -n ${FAKE_STAT_OWNER_PATH-} && ${3-} == \"${FAKE_STAT_OWNER_PATH}\" ]]; then\n"
        "  printf '%s\\n' \"${FAKE_STAT_OWNER_UID}\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ $(/usr/bin/uname -s) == Darwin && ${1-} == -c ]]; then\n"
        "  case ${2-} in\n"
        "    '%u') exec /usr/bin/stat -f '%u' \"$3\" ;;\n"
        "    '%a') exec /usr/bin/stat -f '%OLp' \"$3\" ;;\n"
        "  esac\n"
        "fi\n"
        "exec /usr/bin/stat \"$@\"\n",
    )
    _write_executable(
        fake_bin / "rmdir",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "target=${2-}\n"
        "if [[ -n ${FAKE_RMDIR_FAIL_ONCE_FOR-} "
        "&& ${target} == \"${FAKE_RMDIR_FAIL_ONCE_FOR}\" "
        "&& ! -e ${FAKE_RMDIR_FAILURE_STATE} ]]; then\n"
        "  : > \"${FAKE_RMDIR_FAILURE_STATE}\"\n"
        "  exit 1\n"
        "fi\n"
        "if [[ -x /bin/rmdir ]]; then exec /bin/rmdir \"$@\"; fi\n"
        "exec /usr/bin/rmdir \"$@\"\n",
    )
    fake_conda = _write_executable(
        fake_bin / "conda",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\0' \"$@\" >> \"${CONDA_CAPTURE}\"\n"
        "[[ $1 == run ]]\n"
        "shift\n"
        "if [[ ${1-} == --no-capture-output ]]; then shift; fi\n"
        "[[ $1 == -n && $2 == dmri-repro ]]\n"
        "shift 2\n"
        "[[ $1 == python ]]\n"
        "shift\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
    )
    fake_submitter = _write_executable(
        fake_bin / "CBIG_pbsubmit",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\0' \"$@\" >> \"${CBIG_CAPTURE}\"\n"
        "printf 'fake scheduler stdout\\n'\n"
        "printf 'fake scheduler stderr\\n' >&2\n"
        "if [[ -n ${CBIG_INJECT_DIRECTORY-} ]]; then\n"
        "  rm -f -- \"${CBIG_INJECT_DIRECTORY}\"\n"
        "  mkdir -- \"${CBIG_INJECT_DIRECTORY}\"\n"
        "fi\n"
        "if [[ -n ${CBIG_INJECT_RELATIVE_DIRECTORY-} ]]; then\n"
        "  jobout=\n"
        "  while (( $# )); do\n"
        "    if [[ $1 == -jobout ]]; then jobout=$2; break; fi\n"
        "    shift\n"
        "  done\n"
        "  logs_directory=${jobout%/*}\n"
        "  chain_directory=${logs_directory%/*}\n"
        "  injected=${chain_directory}/${CBIG_INJECT_RELATIVE_DIRECTORY}\n"
        "  rm -f -- \"${injected}\"\n"
        "  mkdir -- \"${injected}\"\n"
        "fi\n"
        "if [[ -n ${CBIG_BLOCK_READY-} ]]; then\n"
        "  printf 'entered\\n' >> \"${CBIG_BLOCK_READY}\"\n"
        "  while [[ ! -f ${CBIG_BLOCK_RELEASE} ]]; do /bin/sleep 0.02; done\n"
        "fi\n"
        "if [[ ${CBIG_EXECUTE_COMMAND:-0} == 1 ]]; then\n"
        "  command_to_run=\n"
        "  while (( $# )); do\n"
        "    if [[ $1 == -cmd ]]; then command_to_run=$2; break; fi\n"
        "    shift\n"
        "  done\n"
        "  if /bin/bash -c \"${command_to_run}\"; then command_status=0; else command_status=$?; fi\n"
        "  printf '%s\\n' \"${command_status}\" >> \"${CBIG_EXECUTION_CAPTURE}\"\n"
        "fi\n"
        "exit \"${CBIG_EXIT:-0}\"\n",
    )
    software_config = tmp_path / "dmri software.sh"
    software_config.write_text(
        "#!/usr/bin/env bash\n"
        + _shell_assignment("CONDA_EXE", str(fake_conda))
        + _shell_assignment("FSLDIR", "/configured/fsl")
        + _shell_assignment("MATLAB_EXECUTABLE", "/configured/matlab")
        + _shell_assignment("DMRI_EXPECTED_FSL_VERSION", "6.0.7.18")
        + _shell_assignment("DMRI_EXPECTED_MATLAB_VERSION", "25.1"),
        encoding="utf-8",
    )
    software_config.chmod(0o600)
    subject_config = tmp_path / "subject config.yaml"
    subject_config.write_text("analysis:\n  noddi_workers: 3\n", encoding="utf-8")
    run_root = tmp_path / "cluster runs"
    cluster_config = tmp_path / "cluster local.sh"
    values = {
        "CBIG_PBSUBMIT": str(fake_submitter),
        "DMRI_SOFTWARE_CONFIG": str(software_config),
        "CLUSTER_RUN_ROOT": str(run_root),
        "TOPUP_WALLTIME": "04:00:00",
        "TOPUP_MEM": "16G",
        "TOPUP_NCPUS": "4",
        "EDDY_WALLTIME": "05:00:00",
        "EDDY_MEM": "24G",
        "EDDY_NCPUS": "6",
        "NODDI_WALLTIME": "25:00:00",
        "NODDI_MEM": "48G",
        "NODDI_NCPUS": "8",
    }
    _write_cluster_config(cluster_config, values)
    return {
        "package": package,
        "subject": subject_config,
        "cluster_config": cluster_config,
        "software_config": software_config,
        "submitter": fake_submitter,
        "conda": fake_conda,
        "fake_bin": fake_bin,
        "run_root": run_root,
        "values": values,
    }


def _write_cluster_config(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        + "".join(_shell_assignment(key, value) for key, value in values.items()),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _environment(tmp_path: Path, *, workers: int | str = 3) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CBIG_CAPTURE": str(tmp_path / "cbig.argv"),
            "CONDA_CAPTURE": str(tmp_path / "conda.argv"),
            "PIPELINE_CAPTURE": str(tmp_path / "pipeline.argv"),
            "PIPELINE_SOFTWARE_CAPTURE": str(tmp_path / "pipeline.software"),
            "CONTEXT_CAPTURE": str(tmp_path / "context.argv"),
            "CONTEXT_SOFTWARE_CAPTURE": str(tmp_path / "context.software"),
            "PATH": f"{tmp_path / 'fake bin'}:{environment['PATH']}",
            "FAKE_CONTEXT_JSON": json.dumps(
                {
                    "noddi_workers": workers,
                    "subject_id": "SUBJECT 001",
                    "subject_output": "/science/output/SUBJECT 001",
                },
                sort_keys=True,
            ),
        }
    )
    return environment


def _run_launcher(
    fixture: dict[str, Path | dict[str, str]],
    environment: dict[str, str],
    *arguments: str,
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    package = fixture["package"]
    assert isinstance(package, Path)
    launcher = package / "scripts" / "cluster" / "submit_subject_chain.sh"
    return subprocess.run(
        [str(launcher), *arguments],
        cwd=working_directory or package,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_launcher_with_closed_stdout(
    fixture: dict[str, Path | dict[str, str]],
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    package = fixture["package"]
    assert isinstance(package, Path)
    launcher = package / "scripts" / "cluster" / "submit_subject_chain.sh"
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        return subprocess.run(
            [str(launcher), *arguments],
            cwd=package,
            env=environment,
            stdout=write_fd,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        os.close(write_fd)


def _nul_arguments(path: Path) -> list[str]:
    return [item.decode() for item in path.read_bytes().split(b"\0") if item]


def _launch_chain(
    fixture: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    environment: dict[str, str],
    start_at: str,
) -> Path:
    subject = fixture["subject"]
    cluster_config = fixture["cluster_config"]
    run_root = fixture["run_root"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(run_root, Path)
    result = _run_launcher(
        fixture,
        environment,
        "--start-at",
        start_at,
        str(subject),
        str(cluster_config),
    )
    assert result.returncode == 0, result.stderr
    chain_dirs = list(run_root.iterdir())
    assert len(chain_dirs) == 1
    for capture_name in ("cbig.argv", "pipeline.argv"):
        capture = tmp_path / capture_name
        if capture.exists():
            capture.unlink()
    return chain_dirs[0]


def _run_worker(
    fixture: dict[str, Path | dict[str, str]],
    environment: dict[str, str],
    group: str,
    chain_dir: Path,
    *,
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    package = fixture["package"]
    subject = fixture["subject"]
    cluster_config = fixture["cluster_config"]
    assert isinstance(package, Path)
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    wrapper = package / "scripts" / "cluster" / f"run_{group}_subject.sh"
    return subprocess.run(
        [str(wrapper), str(subject), str(cluster_config), chain_dir.name],
        cwd=working_directory or package,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_relocated_launcher_and_worker_use_their_physical_repository_root(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    """A relocated process must not substitute its caller's working directory."""
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    run_root = cluster_package["run_root"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(run_root, Path)
    unrelated_cwd = tmp_path / "unrelated caller directory"
    unrelated_cwd.mkdir()
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"

    launch = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
        working_directory=unrelated_cwd,
    )

    assert launch.returncode == 0, launch.stderr
    chain_dirs = list(run_root.iterdir())
    assert len(chain_dirs) == 1
    chain_dir = chain_dirs[0]

    worker = _run_worker(
        cluster_package,
        environment,
        "topup",
        chain_dir,
        working_directory=unrelated_cwd,
    )

    assert worker.returncode == 0, worker.stderr
    assert _nul_arguments(tmp_path / "pipeline.argv") == [
        "--stage-group",
        "topup",
        str(subject),
    ]


@pytest.mark.parametrize("relative_argument", ["subject.yaml", "cluster.local.sh"])
def test_launcher_rejects_relative_config_paths(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    relative_argument: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(subject, Path) and isinstance(cluster_config, Path)
    arguments = [str(subject), str(cluster_config)]
    arguments[0 if relative_argument.startswith("subject") else 1] = relative_argument

    result = _run_launcher(cluster_package, _environment(tmp_path), *arguments)

    assert result.returncode != 0
    assert "absolute" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("missing", ["subject", "cluster_config"])
def test_launcher_rejects_absent_config_files(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    missing: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(subject, Path) and isinstance(cluster_config, Path)
    chosen_subject = tmp_path / "absent subject.yaml" if missing == "subject" else subject
    chosen_cluster = (
        tmp_path / "absent cluster.sh" if missing == "cluster_config" else cluster_config
    )

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(chosen_subject),
        str(chosen_cluster),
    )

    assert result.returncode != 0
    assert "regular file" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize(
    ("key", "bad_value", "message"),
    [
        ("CBIG_PBSUBMIT", "/absolute/path/to/CBIG_pbsubmit", "placeholder"),
        ("DMRI_SOFTWARE_CONFIG", "/absolute/path/to/dmri-rocky9.sh", "placeholder"),
        ("CLUSTER_RUN_ROOT", "/absolute/path/to/cluster-runs", "placeholder"),
        ("TOPUP_WALLTIME", "4 hours", "hh:mm:ss"),
        ("EDDY_MEM", "lots", "memory"),
        ("NODDI_NCPUS", "0", "positive integer"),
        ("TOPUP_MEM", "16G\nunsafe", "control"),
    ],
)
def test_launcher_rejects_invalid_cluster_config_values(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    key: str,
    bad_value: str,
    message: str,
) -> None:
    values = cluster_package["values"]
    cluster_config = cluster_package["cluster_config"]
    subject = cluster_package["subject"]
    assert isinstance(values, dict)
    assert isinstance(cluster_config, Path) and isinstance(subject, Path)
    invalid_values = dict(values)
    invalid_values[key] = bad_value
    _write_cluster_config(cluster_config, invalid_values)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode != 0
    assert message in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


def test_launcher_rejects_missing_cluster_config_key(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    values = cluster_package["values"]
    cluster_config = cluster_package["cluster_config"]
    subject = cluster_package["subject"]
    assert isinstance(values, dict)
    assert isinstance(cluster_config, Path) and isinstance(subject, Path)
    incomplete_values = dict(values)
    del incomplete_values["EDDY_MEM"]
    _write_cluster_config(cluster_config, incomplete_values)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode != 0
    assert "eddy_mem" in result.stderr.lower()
    assert "required" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("config_kind", ["cluster_config", "software_config"])
def test_launcher_rejects_group_or_world_writable_private_config(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    config_kind: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    private_config = cluster_package[config_kind]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(private_config, Path)
    private_config.chmod(0o666)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode == 30
    assert "group- or world-writable" in result.stderr
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("config_kind", ["cluster_config", "software_config"])
def test_launcher_rejects_symlinked_private_config(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    config_kind: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    private_config = cluster_package[config_kind]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(private_config, Path)
    real_config = private_config.with_name(private_config.name + ".real")
    private_config.rename(real_config)
    private_config.symlink_to(real_config)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode == 30
    assert "symlink" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("config_kind", ["cluster_config", "software_config"])
def test_launcher_rejects_private_config_not_owned_by_current_user(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    config_kind: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    private_config = cluster_package[config_kind]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(private_config, Path)
    environment = _environment(tmp_path)
    environment["FAKE_STAT_OWNER_PATH"] = str(private_config)
    environment["FAKE_STAT_OWNER_UID"] = str(os.getuid() + 1)

    result = _run_launcher(
        cluster_package, environment, str(subject), str(cluster_config)
    )

    assert result.returncode == 30
    assert "owned by the current user" in result.stderr
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("config_kind", ["cluster_config", "software_config"])
def test_private_config_exit_cannot_terminate_launcher_successfully(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    config_kind: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    private_config = cluster_package[config_kind]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(private_config, Path)
    private_config.write_text("printf 'private diagnostic\\n'\nexit 0\n", encoding="utf-8")

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode == 30
    assert "private diagnostic" not in result.stdout
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("config_kind", ["cluster_config", "software_config"])
def test_private_config_cannot_resolve_required_value_from_parent_environment(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    config_kind: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    private_config = cluster_package[config_kind]
    submitter = cluster_package["submitter"]
    conda = cluster_package["conda"]
    values = cluster_package["values"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(private_config, Path)
    assert isinstance(submitter, Path)
    assert isinstance(conda, Path)
    assert isinstance(values, dict)
    if config_kind == "cluster_config":
        private_config.write_text(
            "CBIG_PBSUBMIT=\"${PARENT_ONLY}\"\n"
            + "".join(
                _shell_assignment(key, value)
                for key, value in values.items()
                if key != "CBIG_PBSUBMIT"
            ),
            encoding="utf-8",
        )
        inherited_value = str(submitter)
    else:
        private_config.write_text(
            "CONDA_EXE=\"${PARENT_ONLY}\"\n"
            + _shell_assignment("FSLDIR", "/configured/fsl")
            + _shell_assignment("MATLAB_EXECUTABLE", "/configured/matlab")
            + _shell_assignment("DMRI_EXPECTED_FSL_VERSION", "6.0.7.18")
            + _shell_assignment("DMRI_EXPECTED_MATLAB_VERSION", "25.1"),
            encoding="utf-8",
        )
        inherited_value = str(conda)
    environment = _environment(tmp_path)
    environment["PARENT_ONLY"] = inherited_value

    result = _run_launcher(
        cluster_package, environment, str(subject), str(cluster_config)
    )

    assert result.returncode == 30
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("config_kind", ["cluster_config", "software_config"])
def test_private_config_stdout_remains_diagnostic_and_does_not_corrupt_import(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    config_kind: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    private_config = cluster_package[config_kind]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(private_config, Path)
    original = private_config.read_text(encoding="utf-8")
    private_config.write_text(
        "printf 'private diagnostic\\n'\n" + original, encoding="utf-8"
    )

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode == 0, result.stderr
    assert "private diagnostic" not in result.stdout
    assert "private diagnostic" in result.stderr
    assert (tmp_path / "cbig.argv").is_file()


def test_cluster_config_imports_only_fixed_keys_into_launcher(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    with cluster_config.open("a", encoding="utf-8") as handle:
        handle.write("export CBIG_EXIT=77\n")

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cbig.argv").is_file()


@pytest.mark.parametrize("path_key", ["CBIG_PBSUBMIT", "DMRI_SOFTWARE_CONFIG"])
def test_launcher_rejects_missing_configured_files(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    path_key: str,
) -> None:
    values = cluster_package["values"]
    cluster_config = cluster_package["cluster_config"]
    subject = cluster_package["subject"]
    assert isinstance(values, dict)
    assert isinstance(cluster_config, Path) and isinstance(subject, Path)
    invalid_values = dict(values)
    invalid_values[path_key] = str(tmp_path / f"missing {path_key}")
    _write_cluster_config(cluster_config, invalid_values)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode != 0
    assert "regular" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


def test_launcher_rejects_automatic_noddi_workers(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(subject, Path) and isinstance(cluster_config, Path)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path, workers="auto"),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode != 0
    assert "noddi_workers" in result.stderr
    assert "positive integer" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


def test_launcher_rejects_noddi_cpu_request_below_worker_count(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(subject, Path) and isinstance(cluster_config, Path)

    result = _run_launcher(
        cluster_package,
        _environment(tmp_path, workers=9),
        str(subject),
        str(cluster_config),
    )

    assert result.returncode != 0
    assert "noddi_ncpus" in result.stderr.lower()
    assert "noddi_workers" in result.stderr.lower()
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize(
    ("start_at", "walltime", "memory", "ncpus", "wrapper_name"),
    [
        ("topup", "04:00:00", "16G", "4", "run_topup_subject.sh"),
        ("eddy", "05:00:00", "24G", "6", "run_eddy_subject.sh"),
        ("noddi", "25:00:00", "48G", "8", "run_noddi_subject.sh"),
    ],
)
def test_launcher_submits_requested_start_group_with_exact_resources(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    start_at: str,
    walltime: str,
    memory: str,
    ncpus: str,
    wrapper_name: str,
) -> None:
    package = cluster_package["package"]
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    run_root = cluster_package["run_root"]
    assert all(isinstance(item, Path) for item in (package, subject, cluster_config, run_root))
    environment = _environment(tmp_path)

    result = _run_launcher(
        cluster_package,
        environment,
        "--start-at",
        start_at,
        str(subject),
        str(cluster_config),
    )

    assert result.returncode == 0, result.stderr
    chain_dirs = list(run_root.iterdir())
    assert len(chain_dirs) == 1
    chain_dir = chain_dirs[0]
    chain_id = chain_dir.name
    submitted = _nul_arguments(tmp_path / "cbig.argv")
    assert submitted[0:2] == ["-cmd", submitted[1]]
    assert shlex.split(submitted[1]) == [
        str(package / "scripts" / "cluster" / wrapper_name),
        str(subject),
        str(cluster_config),
        chain_id,
    ]
    assert submitted[2:] == [
        "-walltime",
        walltime,
        "-mem",
        memory,
        "-ncpus",
        ncpus,
        "-name",
        f"dmri_{start_at}_{chain_id}",
        "-jobout",
        str(chain_dir / "logs" / f"{start_at}.out"),
        "-joberr",
        str(chain_dir / "logs" / f"{start_at}.err"),
    ]
    assert (chain_dir / "subject_config").read_text(encoding="utf-8") == str(subject) + "\n"
    assert (chain_dir / "cluster_config").read_text(encoding="utf-8") == str(cluster_config) + "\n"
    assert (chain_dir / "start_group").read_text(encoding="utf-8") == start_at + "\n"
    assert (chain_dir / f"{start_at}.submitted").is_file()
    assert (chain_dir / "submissions" / f"{start_at}.stdout").read_text(
        encoding="utf-8"
    ) == "fake scheduler stdout\n"
    assert (chain_dir / "submissions" / f"{start_at}.stderr").read_text(
        encoding="utf-8"
    ) == "fake scheduler stderr\n"
    conda_arguments = _nul_arguments(tmp_path / "conda.argv")
    assert conda_arguments[:5] == [
        "run",
        "--no-capture-output",
        "-n",
        "dmri-repro",
        "python",
    ]


@pytest.mark.parametrize(
    "injected_target",
    [
        "topup.submitted",
        "submissions/topup.stdout",
        "submissions/topup.stderr",
        "submissions/topup.exit_status",
        "status",
    ],
)
def test_launcher_retains_subject_guard_when_accepted_submission_record_fails(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    injected_target: str,
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    run_root = cluster_package["run_root"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(run_root, Path)
    environment = _environment(tmp_path)
    environment["CBIG_INJECT_RELATIVE_DIRECTORY"] = injected_target

    first = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )

    assert first.returncode == 30
    assert "record" in first.stderr.lower()
    first_submission = (tmp_path / "cbig.argv").read_bytes()
    chain_directories = sorted(run_root.glob("chain-*"))
    assert len(chain_directories) == 1
    guard_root = run_root / ".subject-submission-locks"
    guards = list(guard_root.iterdir())
    assert len(guards) == 1
    owner = guards[0] / "owner"
    owner_text = owner.read_text(encoding="utf-8")
    assert f"chain_id={chain_directories[0].name}\n" in owner_text
    assert "start_group=topup\n" in owner_text
    assert f"job_name=dmri_topup_{chain_directories[0].name}\n" in owner_text
    assert "host=" in owner_text
    assert "pid=" in owner_text
    assert "acquired_utc=" in owner_text

    environment.pop("CBIG_INJECT_RELATIVE_DIRECTORY")
    retry = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )

    assert retry.returncode == 30
    assert "subject submission is locked" in retry.stderr.lower()
    assert "reconcile" in retry.stderr.lower()
    assert (tmp_path / "cbig.argv").read_bytes() == first_submission
    assert sorted(run_root.glob("chain-*")) == chain_directories


def test_launcher_releases_subject_guard_after_definite_scheduler_rejection(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    run_root = cluster_package["run_root"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(run_root, Path)
    environment = _environment(tmp_path)
    environment["CBIG_EXIT"] = "73"

    rejected = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )

    assert rejected.returncode == 73
    assert not (run_root / ".subject-submission-locks").exists()
    environment.pop("CBIG_EXIT")
    accepted = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )
    assert accepted.returncode == 0, accepted.stderr
    assert _nul_arguments(tmp_path / "cbig.argv").count("-cmd") == 2


def test_launcher_retains_subject_guard_when_final_success_output_fails(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    run_root = cluster_package["run_root"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(run_root, Path)
    environment = _environment(tmp_path)

    first = _run_launcher_with_closed_stdout(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )

    assert first.returncode != 0
    first_submission = (tmp_path / "cbig.argv").read_bytes()
    guard_root = run_root / ".subject-submission-locks"
    guards = list(guard_root.iterdir())
    assert len(guards) == 1
    retry = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )
    assert retry.returncode == 30
    assert "subject submission is locked" in retry.stderr.lower()
    assert (tmp_path / "cbig.argv").read_bytes() == first_submission
    assert len(list(run_root.glob("chain-*"))) == 1


def test_launcher_does_not_retry_failed_guard_release_from_exit_trap(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    run_root = cluster_package["run_root"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(run_root, Path)
    environment = _environment(tmp_path)
    context = json.loads(environment["FAKE_CONTEXT_JSON"])
    subject_key = hashlib.sha256(context["subject_output"].encode()).hexdigest()
    guard = run_root / ".subject-submission-locks" / f"{subject_key}.lock"
    environment["FAKE_RMDIR_FAIL_ONCE_FOR"] = str(guard)
    environment["FAKE_RMDIR_FAILURE_STATE"] = str(tmp_path / "rmdir.failed")

    first = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )

    assert first.returncode == 30
    assert guard.is_dir()
    first_submission = (tmp_path / "cbig.argv").read_bytes()
    environment.pop("FAKE_RMDIR_FAIL_ONCE_FOR")
    environment.pop("FAKE_RMDIR_FAILURE_STATE")
    retry = _run_launcher(
        cluster_package,
        environment,
        str(subject),
        str(cluster_config),
    )
    assert retry.returncode == 30
    assert "subject submission is locked" in retry.stderr.lower()
    assert (tmp_path / "cbig.argv").read_bytes() == first_submission
    assert len(list(run_root.glob("chain-*"))) == 1


@pytest.mark.parametrize(
    ("group", "successor", "walltime", "memory", "ncpus"),
    [
        ("topup", "eddy", "05:00:00", "24G", "6"),
        ("eddy", "noddi", "25:00:00", "48G", "8"),
        ("noddi", None, None, None, None),
    ],
)
def test_worker_success_records_completion_and_advances_once(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    group: str,
    successor: str | None,
    walltime: str | None,
    memory: str | None,
    ncpus: str | None,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, group)
    package = cluster_package["package"]
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(package, Path)
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)

    result = _run_worker(cluster_package, environment, group, chain_dir)

    assert result.returncode == 0, result.stderr
    assert _nul_arguments(tmp_path / "pipeline.argv") == [
        "--stage-group",
        group,
        str(subject),
    ]
    assert (chain_dir / f"{group}.exit_status").read_text(encoding="utf-8") == "0\n"
    assert (chain_dir / f"{group}.status").read_text(encoding="utf-8") == "complete\n"
    if successor is None:
        assert not (tmp_path / "cbig.argv").exists()
        assert (chain_dir / "status").read_text(encoding="utf-8") == "complete\n"
        return

    submitted = _nul_arguments(tmp_path / "cbig.argv")
    successor_wrapper = package / "scripts" / "cluster" / f"run_{successor}_subject.sh"
    assert shlex.split(submitted[1]) == [
        str(successor_wrapper),
        str(subject),
        str(cluster_config),
        chain_dir.name,
    ]
    assert submitted[2:8] == [
        "-walltime",
        walltime,
        "-mem",
        memory,
        "-ncpus",
        ncpus,
    ]
    assert (chain_dir / f"{successor}.submitted").is_file()
    assert (chain_dir / "status").read_text(encoding="utf-8") == f"{successor}_submitted\n"


@pytest.mark.parametrize("group", ["topup", "eddy", "noddi"])
@pytest.mark.parametrize("pipeline_status", [20, 21, 30, 40, 50])
def test_worker_failure_preserves_status_and_stops_chain(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    group: str,
    pipeline_status: int,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = str(pipeline_status)
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, group)

    result = _run_worker(cluster_package, environment, group, chain_dir)

    assert result.returncode == pipeline_status
    assert (chain_dir / f"{group}.exit_status").read_text(
        encoding="utf-8"
    ) == f"{pipeline_status}\n"
    assert (chain_dir / f"{group}.status").read_text(encoding="utf-8") == "failed\n"
    assert (chain_dir / "status").read_text(encoding="utf-8") == f"{group}_failed\n"
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize(("group", "successor"), [("topup", "eddy"), ("eddy", "noddi")])
def test_worker_successor_submission_failure_is_recorded(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    group: str,
    successor: str,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, group)
    environment["CBIG_EXIT"] = "73"

    result = _run_worker(cluster_package, environment, group, chain_dir)

    assert result.returncode == 73
    assert (chain_dir / f"{group}.exit_status").read_text(encoding="utf-8") == "0\n"
    assert (chain_dir / f"{group}.status").read_text(encoding="utf-8") == "complete\n"
    assert (chain_dir / "status").read_text(encoding="utf-8") == "submission_failed\n"
    assert (chain_dir / "submission_failed").read_text(
        encoding="utf-8"
    ) == f"{successor}\n"
    assert (chain_dir / "submissions" / f"{successor}.exit_status").read_text(
        encoding="utf-8"
    ) == "73\n"
    assert not (chain_dir / f"{successor}.submitted").exists()


@pytest.mark.parametrize(("group", "successor"), [("topup", "eddy"), ("eddy", "noddi")])
def test_worker_duplicate_success_does_not_resubmit_successor(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    group: str,
    successor: str,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, group)

    first = _run_worker(cluster_package, environment, group, chain_dir)
    first_submission = (tmp_path / "cbig.argv").read_bytes()
    second = _run_worker(cluster_package, environment, group, chain_dir)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "cbig.argv").read_bytes() == first_submission
    assert (chain_dir / f"{successor}.submitted").is_file()


def test_concurrent_successful_workers_contend_on_one_real_successor_lock(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, "topup")
    ready = tmp_path / "submitter.ready"
    release = tmp_path / "submitter.release"
    environment["CBIG_BLOCK_READY"] = str(ready)
    environment["CBIG_BLOCK_RELEASE"] = str(release)
    package = cluster_package["package"]
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(package, Path)
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    command = [
        str(package / "scripts" / "cluster" / "run_topup_subject.sh"),
        str(subject),
        str(cluster_config),
        chain_dir.name,
    ]
    first = subprocess.Popen(
        command,
        cwd=package,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10.0
        while (
            not ready.exists()
            and first.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        if not ready.exists():
            pytest.fail(
                "first worker never entered blocking submitter: "
                f"rc={first.poll()}"
            )

        contender = subprocess.run(
            command,
            cwd=package,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        assert contender.returncode == 30
        assert "successor submission is locked" in contender.stderr
        owner = chain_dir / ".eddy.submission.lock" / "owner"
        owner_text = owner.read_text(encoding="utf-8")
        assert f"chain_id={chain_dir.name}\n" in owner_text
        assert "source_group=topup\n" in owner_text
        assert "successor=eddy\n" in owner_text
        assert f"job_name=dmri_eddy_{chain_dir.name}\n" in owner_text
        assert "host=" in owner_text
        assert "pid=" in owner_text
        assert "acquired_utc=" in owner_text
    finally:
        release.write_text("release\n", encoding="utf-8")
        try:
            first_stdout, first_stderr = first.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            first.kill()
            first_stdout, first_stderr = first.communicate(timeout=5)

    assert first.returncode == 0, (first_stdout, first_stderr)
    assert _nul_arguments(tmp_path / "cbig.argv").count("-cmd") == 1
    assert (chain_dir / "eddy.submitted").is_file()
    assert not (chain_dir / ".eddy.submission.lock").exists()


@pytest.mark.parametrize(
    ("relative_target", "target_kind"),
    [
        (Path("submissions/eddy.argv"), "symlink"),
        (Path("submissions/eddy.exit_status"), "fifo"),
        (Path("eddy.submitted"), "directory"),
    ],
)
def test_successor_submission_rejects_nonregular_record_targets_before_scheduler(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    relative_target: Path,
    target_kind: str,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, "topup")
    target = chain_dir / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside record"
    outside.write_text("preserve\n", encoding="utf-8")
    if target_kind == "symlink":
        target.symlink_to(outside)
    elif target_kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    result = _run_worker(cluster_package, environment, "topup", chain_dir)

    assert result.returncode == 30
    assert "record" in result.stderr.lower() or "symbolic link" in result.stderr.lower()
    assert outside.read_text(encoding="utf-8") == "preserve\n"
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize(
    ("injected_target", "marker_must_exist", "lock_must_remain"),
    [
        (Path("submissions/eddy.exit_status"), True, True),
        (Path("eddy.submitted"), False, True),
        (Path("status"), True, True),
    ],
)
def test_successor_submission_propagates_record_failure_after_scheduler_acceptance(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    injected_target: Path,
    marker_must_exist: bool,
    lock_must_remain: bool,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, "topup")
    environment["CBIG_INJECT_DIRECTORY"] = str(chain_dir / injected_target)

    result = _run_worker(cluster_package, environment, "topup", chain_dir)

    assert result.returncode == 30
    assert "record" in result.stderr.lower()
    assert (tmp_path / "cbig.argv").is_file()
    assert (chain_dir / "eddy.submitted").is_file() is marker_must_exist
    lock = chain_dir / ".eddy.submission.lock"
    assert lock.is_dir() is lock_must_remain
    if lock_must_remain:
        assert (lock / "owner").is_file()
        first_submission = (tmp_path / "cbig.argv").read_bytes()
        (chain_dir / injected_target).rmdir()
        environment.pop("CBIG_INJECT_DIRECTORY")
        retry = _run_worker(cluster_package, environment, "topup", chain_dir)
        assert retry.returncode == 30
        assert "successor submission is locked" in retry.stderr
        assert (tmp_path / "cbig.argv").read_bytes() == first_submission


@pytest.mark.parametrize("replaced_directory", ["chain", "logs", "submissions"])
def test_worker_rejects_chain_directories_resolving_outside_run_root(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    replaced_directory: str,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, "topup")
    if replaced_directory == "chain":
        replaced_path = chain_dir
        moved_path = tmp_path / "outside chain"
    else:
        replaced_path = chain_dir / replaced_directory
        moved_path = tmp_path / f"outside {replaced_directory}"
    replaced_path.rename(moved_path)
    replaced_path.symlink_to(moved_path, target_is_directory=True)

    result = _run_worker(cluster_package, environment, "topup", chain_dir)

    assert result.returncode != 0
    assert "cluster_run_root" in result.stderr.lower()
    assert not (tmp_path / "pipeline.argv").exists()
    assert not (tmp_path / "cbig.argv").exists()


@pytest.mark.parametrize("group", ["topup", "eddy", "noddi"])
def test_worker_rejects_mutated_noddi_cpu_request_below_immutable_worker_count(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    group: str,
) -> None:
    environment = _environment(tmp_path, workers=3)
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, group)
    conda_capture = tmp_path / "conda.argv"
    context_calls_before_worker = conda_capture.read_bytes()
    values = cluster_package["values"]
    cluster_config = cluster_package["cluster_config"]
    assert isinstance(values, dict)
    assert isinstance(cluster_config, Path)
    mutated_values = dict(values)
    mutated_values["NODDI_NCPUS"] = "2"
    _write_cluster_config(cluster_config, mutated_values)
    environment["FAKE_CONTEXT_JSON"] = json.dumps(
        {
            "noddi_workers": 1,
            "subject_id": "MUTATED",
            "subject_output": "/mutated/output",
        },
        sort_keys=True,
    )

    result = _run_worker(cluster_package, environment, group, chain_dir)

    assert result.returncode != 0
    assert "noddi_ncpus" in result.stderr.lower()
    assert "noddi_workers" in result.stderr.lower()
    assert not (tmp_path / "pipeline.argv").exists()
    assert not (tmp_path / "cbig.argv").exists()
    assert conda_capture.read_bytes() == context_calls_before_worker
    assert not (chain_dir / f"{group}.started_at").exists()


@pytest.mark.parametrize("group", ["topup", "eddy", "noddi"])
@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("subject_id", "MUTATED SUBJECT"),
        ("subject_output", "/mutated/output"),
        ("noddi_workers", 7),
    ],
)
def test_worker_revalidates_live_context_against_immutable_chain_identity(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    group: str,
    field: str,
    mutated_value: str | int,
) -> None:
    environment = _environment(tmp_path, workers=3)
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, group)
    mutated_context: dict[str, str | int] = {
        "noddi_workers": 3,
        "subject_id": "SUBJECT 001",
        "subject_output": "/science/output/SUBJECT 001",
    }
    mutated_context[field] = mutated_value
    environment["FAKE_CONTEXT_JSON"] = json.dumps(mutated_context, sort_keys=True)

    result = _run_worker(cluster_package, environment, group, chain_dir)

    assert result.returncode == 30
    assert field in result.stderr
    assert "immutable chain" in result.stderr.lower()
    assert not (tmp_path / "pipeline.argv").exists()
    assert not (tmp_path / "cbig.argv").exists()
    assert not (chain_dir / f"{group}.started_at").exists()


@pytest.mark.parametrize(
    ("advanced_status", "downstream_groups", "downstream_failure"),
    [
        ("eddy_failed", ("eddy",), 50),
        ("noddi_submitted", ("eddy",), None),
        ("complete", ("eddy", "noddi"), None),
    ],
)
def test_reentered_topup_worker_preserves_later_or_terminal_global_status(
    cluster_package: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    advanced_status: str,
    downstream_groups: tuple[str, ...],
    downstream_failure: int | None,
) -> None:
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "0"
    chain_dir = _launch_chain(cluster_package, tmp_path, environment, "topup")
    topup = _run_worker(cluster_package, environment, "topup", chain_dir)
    assert topup.returncode == 0, topup.stderr
    for downstream_group in downstream_groups:
        environment["FAKE_PIPELINE_STATUS"] = str(downstream_failure or 0)
        downstream = _run_worker(
            cluster_package, environment, downstream_group, chain_dir
        )
        assert downstream.returncode == (downstream_failure or 0), downstream.stderr
    assert (chain_dir / "status").read_text(encoding="utf-8") == advanced_status + "\n"
    environment["FAKE_PIPELINE_STATUS"] = "0"

    retried = _run_worker(cluster_package, environment, "topup", chain_dir)

    assert retried.returncode == 0, retried.stderr
    assert (chain_dir / "status").read_text(encoding="utf-8") == advanced_status + "\n"


def test_context_preflight_receives_exact_subject_argv_and_software_environment(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    environment = _environment(tmp_path)
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    software_config = cluster_package["software_config"]
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(software_config, Path)

    result = _run_launcher(
        cluster_package, environment, str(subject), str(cluster_config)
    )

    assert result.returncode == 0, result.stderr
    assert _nul_arguments(tmp_path / "context.argv") == [
        "--print-cluster-context",
        str(subject),
    ]
    assert _nul_arguments(tmp_path / "context.software") == [str(software_config)]


def test_generated_scheduler_command_executes_difficult_valid_paths_exactly(
    cluster_package: dict[str, Path | dict[str, str]], tmp_path: Path
) -> None:
    package = cluster_package["package"]
    subject = cluster_package["subject"]
    cluster_config = cluster_package["cluster_config"]
    software_config = cluster_package["software_config"]
    assert isinstance(package, Path)
    assert isinstance(subject, Path)
    assert isinstance(cluster_config, Path)
    assert isinstance(software_config, Path)
    injection_sentinel = package / "cluster-command-injected"
    difficult_subject = (
        tmp_path
        / "subject $(touch cluster-command-injected) []$literal;quote'*? config.yaml"
    )
    difficult_cluster = tmp_path / "cluster []$literal;quote'*? config.sh"
    subject.rename(difficult_subject)
    cluster_config.rename(difficult_cluster)
    cluster_package["subject"] = difficult_subject
    cluster_package["cluster_config"] = difficult_cluster
    environment = _environment(tmp_path)
    environment["FAKE_PIPELINE_STATUS"] = "21"
    environment["CBIG_EXECUTE_COMMAND"] = "1"
    environment["CBIG_EXECUTION_CAPTURE"] = str(tmp_path / "executed.status")

    result = _run_launcher(
        cluster_package,
        environment,
        str(difficult_subject),
        str(difficult_cluster),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "executed.status").read_text(encoding="utf-8") == "21\n"
    assert _nul_arguments(tmp_path / "pipeline.argv") == [
        "--stage-group",
        "topup",
        str(difficult_subject),
    ]
    assert _nul_arguments(tmp_path / "pipeline.software") == [str(software_config)]
    assert not injection_sentinel.exists()
