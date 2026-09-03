from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

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

    fake_pipeline = _write_executable(
        package / "run_pipeline.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ ${1-} == --print-cluster-context ]]; then\n"
        "  printf '%s\\n' \"${FAKE_CONTEXT_JSON}\"\n"
        "  exit \"${FAKE_CONTEXT_STATUS:-0}\"\n"
        "fi\n"
        "printf '%s\\0' \"$@\" >> \"${PIPELINE_CAPTURE}\"\n"
        "exit \"${FAKE_PIPELINE_STATUS:-0}\"\n",
    )
    assert fake_pipeline.is_file()

    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
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
        "exit \"${CBIG_EXIT:-0}\"\n",
    )
    software_config = tmp_path / "dmri software.sh"
    software_config.write_text(
        "#!/usr/bin/env bash\n"
        + _shell_assignment("CONDA_EXE", str(fake_conda)),
        encoding="utf-8",
    )
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
        "run_root": run_root,
        "values": values,
    }


def _write_cluster_config(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        + "".join(_shell_assignment(key, value) for key, value in values.items()),
        encoding="utf-8",
    )


def _environment(tmp_path: Path, *, workers: int | str = 3) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CBIG_CAPTURE": str(tmp_path / "cbig.argv"),
            "CONDA_CAPTURE": str(tmp_path / "conda.argv"),
            "PIPELINE_CAPTURE": str(tmp_path / "pipeline.argv"),
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
) -> subprocess.CompletedProcess[str]:
    package = fixture["package"]
    assert isinstance(package, Path)
    launcher = package / "scripts" / "cluster" / "submit_subject_chain.sh"
    return subprocess.run(
        [str(launcher), *arguments],
        cwd=package,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


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
        cwd=package,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


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
