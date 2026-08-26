# Rocky Linux and VS Code Remote Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the macOS-only deployment contract with a fail-closed Rocky Linux 9.7 x86_64 deployment that is edited from macOS through VS Code Remote SSH while all data and computation remain on the server.

**Architecture:** Large and licensed applications remain independently installed on the Rocky server. One operator-owned software configuration supplies exact Conda, FSL, and MATLAB paths to both the setup and runtime wrappers; Python keeps using the pinned Conda environment. VS Code Remote SSH edits the repository in place, while platform checks, dependency probes, pipeline execution, and long-running `tmux` sessions all execute on Rocky.

**Tech Stack:** Bash, Python 3.11, pytest, Conda/Miniforge, FSL, MATLAB with Optimization Toolbox and MEX, VS Code Remote SSH, tmux.

## Global Constraints

- The only supported execution platform is Rocky Linux 9.7 on x86_64.
- The macOS computer is a VS Code and SSH client only; it must not run pipeline stages.
- Source, MRI inputs, work files, logs, and outputs remain on the Rocky server.
- The deployment runs directly on the server, without Slurm, PBS, Docker, Apptainer, or Singularity.
- FSL, MATLAB, Optimization Toolbox, the MATLAB licence, and the C compiler are not installed or licensed by this package.
- SSH usernames, hostnames, passwords, private keys, licence secrets, and machine-specific paths must not enter the repository.
- Missing, mismatched, or incomplete dependencies fail before scientific stages start.
- The scientific algorithms, stage order, exit-code contract, resource hashes, and resume semantics remain unchanged.
- CPU EDDY remains the supported EDDY mode; GPU/CUDA EDDY is out of scope.
- The existing Darwin atomic-rename branch may remain only for pre-migration unit-test portability; public setup and runtime entry points reject macOS.
- Long-running server runs use `tmux`; MRI processing must not run through SSHFS.

## File map

- `config/software.rocky.example.sh`: public, placeholder-only software path and exact-version contract.
- `scripts/rocky_environment.sh`: non-executable shared loader and Rocky platform gate used by both shell entry points.
- `setup_rocky.sh`: Rocky platform gate, software configuration loader, environment installation, and fail-closed preflight.
- `run_pipeline.sh`: runtime configuration loader and Conda launcher; rejects unsupported hosts before launching Python.
- `src/dmri_pipeline/orchestrator.py`: Linux physical-memory discovery for automatic NODDI worker selection.
- `src/dmri_pipeline/noddi.py`: MATLAB discovery precedence for explicit YAML, server environment, and server `PATH`.
- `src/dmri_pipeline/package_audit.py`: public-package executable allowlist.
- `tests/test_wrappers.py`: shell wrapper, configuration, platform, external-version, and MEX preflight tests.
- `tests/test_orchestrator.py`: `/proc/meminfo` parsing and error propagation tests.
- `tests/test_noddi.py`: Linux MATLAB discovery tests.
- `tests/test_package_audit.py`: exported filenames, required documentation, public contract, and workspace safety tests.
- `tests/test_state.py`: expected breaking-release package version.
- `README.md`, `docs/INSTALL_ROCKY.md`, `docs/TROUBLESHOOTING.md`: Rocky public usage contract.
- `docs/REMOTE_VSCODE.md`: Mac-to-Rocky Remote SSH and `tmux` workflow.
- `.vscode/extensions.json`: non-secret extension recommendations.
- `VERSION`, `pyproject.toml`: breaking-release metadata.

---

### Task 1: Rocky software configuration and shell entry points

**Files:**
- Create: `config/software.rocky.example.sh`
- Create: `scripts/rocky_environment.sh`
- Rename: `setup_macos.sh` to `setup_rocky.sh`
- Modify: `setup_rocky.sh`
- Modify: `run_pipeline.sh`
- Modify: `tests/test_wrappers.py`

**Interfaces:**
- Consumes: absolute path in `DMRI_SOFTWARE_CONFIG`; optional test-only `DMRI_OS_RELEASE_FILE`, defaulting to `/etc/os-release`.
- Produces: shared `load_software_config()` and `check_rocky_platform()` shell functions; exported `CONDA_EXE`, `FSLDIR`, `MATLAB_EXECUTABLE`, `DMRI_EXPECTED_FSL_VERSION`, and `DMRI_EXPECTED_MATLAB_VERSION`; executable `setup_rocky.sh [--check]`; unchanged `run_pipeline.sh PIPELINE_ARGS...` forwarding after platform and environment validation.

- [ ] **Step 1: Write failing wrapper and configuration tests**

Add helpers that write a safe private software file and a synthetic Rocky release file. Make the existing fake MATLAB report Linux `mexa64`, create `${FSLDIR}/etc/fslversion`, and set exact expected versions.

```python
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
```

Add these tests and update every existing setup invocation to use `setup_rocky.sh` plus the private configuration:

```python
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
```

The fake `uname` in the test `PATH` must return `Linux` for `-s` and `x86_64` for `-m`; do not add a production bypass for architecture checking.

- [ ] **Step 2: Run the wrapper tests and confirm the expected failures**

Run:

```bash
python -m pytest tests/test_wrappers.py -q
```

Expected: failures name the missing `setup_rocky.sh`, missing software configuration loader, and missing Rocky/version checks.

- [ ] **Step 3: Add the public software configuration example**

Create `config/software.rocky.example.sh` as a non-executable, placeholder-only file:

```bash
#!/usr/bin/env bash

export CONDA_EXE="/opt/miniforge3/bin/conda"
export FSLDIR="/apps/fsl/REPLACE_WITH_FSL_VERSION"
export MATLAB_EXECUTABLE="/apps/matlab/REPLACE_WITH_MATLAB_RELEASE/bin/matlab"
export DMRI_EXPECTED_FSL_VERSION="REPLACE_WITH_EXACT_FSL_VERSION"
export DMRI_EXPECTED_MATLAB_VERSION="REPLACE_WITH_EXACT_MATLAB_VERSION"
```

Do not add this example to `.bashrc`, do not include an actual server hostname or username, and do not make it executable.

- [ ] **Step 4: Implement one shared strict software-configuration loader**

Create non-executable `scripts/rocky_environment.sh`. Both wrappers define their own `fail()` first, source this library by a relocation-safe path, and call the same function:

```bash
load_software_config() {
    local config_path=${DMRI_SOFTWARE_CONFIG:-}
    [[ -n "${config_path}" ]] \
        || fail "DMRI_SOFTWARE_CONFIG must name the private server software configuration"
    case "${config_path}" in
        /*) ;;
        *) fail "DMRI_SOFTWARE_CONFIG must be an absolute path" ;;
    esac
    [[ -f "${config_path}" && ! -L "${config_path}" && -r "${config_path}" ]] \
        || fail "DMRI_SOFTWARE_CONFIG must be a readable regular file, not a symlink"
    [[ "$(stat -c '%u' "${config_path}")" == "$(id -u)" ]] \
        || fail "DMRI_SOFTWARE_CONFIG must be owned by the current user"
    local mode
    mode=$(stat -c '%a' "${config_path}")
    (( (8#${mode} & 8#022) == 0 )) \
        || fail "DMRI_SOFTWARE_CONFIG must not be group- or world-writable"
    # shellcheck disable=SC1090
    source "${config_path}"
    for name in CONDA_EXE FSLDIR MATLAB_EXECUTABLE \
        DMRI_EXPECTED_FSL_VERSION DMRI_EXPECTED_MATLAB_VERSION
    do
        [[ -n "${!name:-}" ]] || fail "software configuration is missing ${name}"
        export "${name}"
    done
}
```

Add a wrapper test that changes a synthetic configuration to mode `0666` and expects exit 30 with `group- or world-writable`. Temporary test files already satisfy the current-user ownership rule; do not weaken the rule to support a shared mutable configuration.

In `run_pipeline.sh`, define `fail()` to print `ERROR:` and exit 30, call the shared platform gate and `load_software_config` before Conda discovery, and use the configured `CONDA_EXE`. Preserve exact argument forwarding and the child pipeline exit code.

- [ ] **Step 5: Replace the macOS platform check with a Rocky 9.7 x86_64 gate**

Rename the setup file with `git mv` and keep its executable bit. Put parsing that does not execute `/etc/os-release` as shell code in `scripts/rocky_environment.sh`:

```bash
os_release_value() {
    local key=$1 file=$2 value
    value=$(sed -n "s/^${key}=//p" "${file}" | head -n 1)
    value=${value#\"}
    value=${value%\"}
    printf '%s\n' "${value}"
}

check_rocky_platform() {
    local release_file=${DMRI_OS_RELEASE_FILE:-/etc/os-release}
    local os_name architecture os_id version_id
    [[ -f "${release_file}" ]] || fail "cannot read OS release file: ${release_file}"
    os_name=$(uname -s)
    architecture=$(uname -m)
    os_id=$(os_release_value ID "${release_file}")
    version_id=$(os_release_value VERSION_ID "${release_file}")
    [[ "${os_name}" == "Linux" && "${os_id}" == "rocky" \
        && "${version_id}" == "9.7" ]] \
        || fail "Rocky Linux 9.7 is required"
    [[ "${architecture}" == "x86_64" ]] \
        || fail "x86_64 is required; found ${architecture}"
    echo "OK: Rocky Linux ${version_id} ${architecture}"
}
```

Call `check_rocky_platform` and then `load_software_config` at the start of both public wrappers. Remove `/Applications` lookup from `check_matlab`, require configured absolute executables, require `mexa64`, compare the probed MATLAB version with `DMRI_EXPECTED_MATLAB_VERSION`, and compare `${FSLDIR}/etc/fslversion` with `DMRI_EXPECTED_FSL_VERSION`.

- [ ] **Step 6: Run wrapper tests to green**

Run:

```bash
bash -n setup_rocky.sh run_pipeline.sh scripts/rocky_environment.sh \
  config/software.rocky.example.sh
python -m pytest tests/test_wrappers.py -q
```

Expected: shell syntax succeeds and all wrapper tests pass on the Mac test host using only synthetic Rocky/tool fixtures.

- [ ] **Step 7: Commit the shell contract**

```bash
git add -- config/software.rocky.example.sh scripts/rocky_environment.sh \
  setup_rocky.sh run_pipeline.sh tests/test_wrappers.py
git commit -m "feat: add Rocky server software contract"
```

---

### Task 2: Linux memory discovery for NODDI workers

**Files:**
- Modify: `src/dmri_pipeline/orchestrator.py:1281-1307`
- Modify: `tests/test_orchestrator.py:176-184`
- Modify: `tests/test_orchestrator.py:1835-1845`

**Interfaces:**
- Consumes: UTF-8 text from `/proc/meminfo` with exactly one positive `MemTotal: N kB` record.
- Produces: `_parse_memtotal_kib(text: str) -> int` and `_installed_memory_gib(meminfo_path: Path = Path("/proc/meminfo")) -> float`.

- [ ] **Step 1: Replace the sysctl unit test with parser and file tests**

Add explicit success and rejection coverage:

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MemTotal:       16777216 kB\nMemFree: 1 kB\n", 16777216),
        ("MemFree: 1 kB\nMemTotal: 8388608 kB\n", 8388608),
    ],
)
def test_parse_memtotal_kib(text: str, expected: int) -> None:
    assert orchestrator._parse_memtotal_kib(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MemFree: 1 kB\n",
        "MemTotal: 0 kB\n",
        "MemTotal: -1 kB\n",
        "MemTotal: unknown kB\n",
        "MemTotal: 1 MB\n",
        "MemTotal: 1 kB\nMemTotal: 2 kB\n",
    ],
)
def test_parse_memtotal_kib_rejects_malformed_values(text: str) -> None:
    with pytest.raises(NODDIError, match="MemTotal"):
        orchestrator._parse_memtotal_kib(text)


def test_installed_memory_reads_proc_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16777216 kB\n", encoding="utf-8")
    assert orchestrator._installed_memory_gib(meminfo) == 16.0
```

Update the dry-run failure test to expect `/proc/meminfo` or `MemTotal`, not `sysctl`.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
python -m pytest \
  tests/test_orchestrator.py::test_parse_memtotal_kib \
  tests/test_orchestrator.py::test_parse_memtotal_kib_rejects_malformed_values \
  tests/test_orchestrator.py::test_installed_memory_reads_proc_meminfo -q
```

Expected: FAIL because the parser does not exist and `_installed_memory_gib` still invokes macOS `sysctl`.

- [ ] **Step 3: Implement strict `/proc/meminfo` parsing**

Replace the command-runner implementation with:

```python
def _parse_memtotal_kib(text: str) -> int:
    matches = re.findall(r"(?m)^MemTotal:\s+([0-9]+)\s+kB\s*$", text)
    if len(matches) != 1:
        raise NODDIError("/proc/meminfo must contain exactly one MemTotal value in kB")
    value = int(matches[0])
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
```

Import `re` if it is not already imported. Do not consult swap, free memory, Mac `sysctl`, Slurm, or cgroups in this release.

- [ ] **Step 4: Run memory and orchestrator tests**

```bash
python -m pytest tests/test_orchestrator.py -q
```

Expected: all orchestrator tests pass and no assertion refers to `hw.memsize`.

- [ ] **Step 5: Commit Linux memory discovery**

```bash
git add -- src/dmri_pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "fix: read NODDI memory from procfs"
```

---

### Task 3: Server-side MATLAB discovery

**Files:**
- Modify: `src/dmri_pipeline/noddi.py:220-248`
- Modify: `tests/test_noddi.py:205-280`

**Interfaces:**
- Consumes: `PipelineConfig.matlab_executable`, then `MATLAB_EXECUTABLE`, then `shutil.which("matlab")`.
- Produces: the existing `MATLABInstallation` result after the unchanged capability probe; no `/Applications` fallback.

- [ ] **Step 1: Write failing environment precedence tests**

```python
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
```

Also change Linux-oriented fixtures from `mexmaca64` to `mexa64`; keep isolated parser tests free to accept any syntactically valid MEX extension.

- [ ] **Step 2: Run the new discovery tests and confirm failure**

```bash
python -m pytest \
  tests/test_noddi.py::test_discovery_uses_matlab_executable_environment_before_path \
  tests/test_noddi.py::test_invalid_matlab_environment_does_not_fall_back -q
```

Expected: FAIL because `MATLAB_EXECUTABLE` is not read.

- [ ] **Step 3: Implement Linux discovery precedence**

Replace the macOS fallback block with explicit source tracking:

```python
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
```

Update the docstring to match this precedence. Do not scan `/Applications`, `/usr/local`, or version glob patterns; the server configuration is authoritative.

- [ ] **Step 4: Run NODDI tests**

```bash
python -m pytest tests/test_noddi.py -q
```

Expected: all NODDI tests pass.

- [ ] **Step 5: Commit MATLAB discovery**

```bash
git add -- src/dmri_pipeline/noddi.py tests/test_noddi.py
git commit -m "fix: discover MATLAB from Rocky server configuration"
```

---

### Task 4: Rocky public package contract and breaking-release metadata

**Files:**
- Rename: `docs/INSTALL_MACOS.md` to `docs/INSTALL_ROCKY.md`
- Modify: `README.md`
- Modify: `docs/INSTALL_ROCKY.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `pyproject.toml`
- Modify: `VERSION`
- Modify: `src/dmri_pipeline/package_audit.py:17-20`
- Modify: `tests/test_package_audit.py:19-27,88-105,465-510`
- Modify: `tests/test_state.py:135`

**Interfaces:**
- Consumes: `setup_rocky.sh`, `config/software.rocky.example.sh`, and `DMRI_SOFTWARE_CONFIG` from Task 1.
- Produces: public version `2.0.0`; executable allowlist `{"run_pipeline.sh", "setup_rocky.sh"}`; required document `docs/INSTALL_ROCKY.md`; Rocky-only quick start and troubleshooting contract.

- [ ] **Step 1: Write failing public-contract tests**

Change constants and assertions first:

```python
REQUIRED_DOCUMENTS = (
    "README.md",
    "docs/INSTALL_ROCKY.md",
    "docs/INPUTS.md",
    "docs/PIPELINE.md",
    "docs/OUTPUTS.md",
    "docs/QC_AND_EXCLUSION.md",
    "docs/TROUBLESHOOTING.md",
)
```

```python
assert audit.executables == ["run_pipeline.sh", "setup_rocky.sh"]
```

Replace the public README requirements with Rocky and server software-configuration terms, and assert quick-start commands exactly:

```python
assert commands == [
    "cp config/subject.example.yaml config/subject.yaml",
    "export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh",
    "./setup_rocky.sh",
    "./run_pipeline.sh config/subject.yaml",
]
```

Update the test's command extraction to include lines beginning with `export ` as well as `cp ` and `./`, so the software configuration command is part of the asserted public contract.

Add metadata consistency and scoped macOS-removal tests:

```python
def test_release_metadata_is_rocky_only_and_consistent() -> None:
    assert (PACKAGE_ROOT / "VERSION").read_text(encoding="utf-8").strip() == "2.0.0"
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.0.0"' in pyproject
    for relative in (
        "README.md",
        "docs/INSTALL_ROCKY.md",
        "docs/TROUBLESHOOTING.md",
        "pyproject.toml",
    ):
        text = (PACKAGE_ROOT / relative).read_text(encoding="utf-8").lower()
        assert "macos" not in text, relative
        assert "setup_macos.sh" not in text, relative
```

Update the stage-record expected package version from `1.0.0` to `2.0.0`.

- [ ] **Step 2: Run public-contract tests and confirm failure**

```bash
python -m pytest \
  tests/test_package_audit.py::test_exported_package_audit_is_clean_and_contains_only_the_historical_atlas \
  tests/test_package_audit.py::test_required_documentation_covers_the_public_contract \
  tests/test_package_audit.py::test_release_metadata_is_rocky_only_and_consistent \
  tests/test_state.py::test_completion_record_is_deterministic_and_parseable -q
```

Expected: FAIL on old filenames, old allowlist, macOS wording, and version `1.0.0`.

- [ ] **Step 3: Update audit and release metadata**

Set:

```python
_ALLOWED_EXECUTABLES = frozenset({"run_pipeline.sh", "setup_rocky.sh"})
```

Set `VERSION` and `[project].version` to `2.0.0`; change the project description to `Patient-free, reproducible Rocky Linux diffusion MRI pipeline`.

- [ ] **Step 4: Rewrite installation and troubleshooting documentation**

Rename the installation document with `git mv`. Document these exact boundaries:

- Rocky Linux 9.7 x86_64 only;
- copy the example software configuration to a private absolute server path;
- edit all five required variables;
- export `DMRI_SOFTWARE_CONFIG` in the current server shell;
- run `./setup_rocky.sh` and `./setup_rocky.sh --check`;
- the setup script verifies but does not install or license FSL/MATLAB/compiler dependencies;
- MATLAB MEX output must be `mexa64`;
- `tools.fsldir` and `tools.matlab_executable` in a subject YAML remain higher-precedence subject-specific overrides;
- CPU EDDY only;
- resolve dependency errors before rerunning.

Rewrite README headings, quick start, supported system, free-disk check command, and documentation links. Do not claim that VS Code or Remote SSH is required to execute the pipeline; it is the supported operator workflow, while the runtime requirement is Rocky.

- [ ] **Step 5: Run audit, state, and documentation tests**

```bash
python -m pytest tests/test_package_audit.py tests/test_state.py -q
```

Expected: all tests pass, package audit sees only the two intended executable scripts, and the historical atlas hash is unchanged.

- [ ] **Step 6: Commit the Rocky public contract**

```bash
git add -- README.md docs/INSTALL_ROCKY.md docs/TROUBLESHOOTING.md \
  pyproject.toml VERSION src/dmri_pipeline/package_audit.py \
  tests/test_package_audit.py tests/test_state.py
git commit -m "docs: publish Rocky-only package contract"
```

---

### Task 5: VS Code Remote SSH and tmux operator workflow

**Files:**
- Create: `.vscode/extensions.json`
- Create: `docs/REMOTE_VSCODE.md`
- Modify: `README.md`
- Modify: `tests/test_package_audit.py`

**Interfaces:**
- Consumes: an already working Mac OpenSSH connection and the server repository path; `DMRI_SOFTWARE_CONFIG` from Task 1.
- Produces: non-secret VS Code recommendations and exact instructions for editing server files, verifying the remote context, selecting the server interpreter, and protecting long runs with `tmux`.

- [ ] **Step 1: Add failing workspace-safety tests**

```python
def test_vscode_recommendations_are_valid_and_non_secret() -> None:
    payload = json.loads(
        (PACKAGE_ROOT / ".vscode" / "extensions.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "recommendations": [
            "ms-vscode-remote.remote-ssh",
            "ms-python.python",
            "ms-python.vscode-pylance",
        ]
    }
    rendered = json.dumps(payload).lower()
    for forbidden in ("hostname", "identityfile", "password", "/users/", "/home/"):
        assert forbidden not in rendered


def test_remote_vscode_document_keeps_compute_on_server() -> None:
    text = (PACKAGE_ROOT / "docs" / "REMOTE_VSCODE.md").read_text(
        encoding="utf-8"
    ).lower()
    for required in (
        "remote - ssh",
        "ssh dmri-rocky",
        "open folder",
        "uname -m",
        "dmri_software_config",
        "tmux new -s",
        "tmux attach",
        "do not use sshfs",
    ):
        assert required in text, required
```

- [ ] **Step 2: Run the new tests and confirm missing-file failures**

```bash
python -m pytest \
  tests/test_package_audit.py::test_vscode_recommendations_are_valid_and_non_secret \
  tests/test_package_audit.py::test_remote_vscode_document_keeps_compute_on_server -q
```

Expected: FAIL because the workspace and remote guide do not exist.

- [ ] **Step 3: Add extension recommendations**

Create `.vscode/extensions.json` exactly as tested:

```json
{
  "recommendations": [
    "ms-vscode-remote.remote-ssh",
    "ms-python.python",
    "ms-python.vscode-pylance"
  ]
}
```

Do not add `.vscode/settings.json` with a fixed interpreter path; the Conda installation location is operator-specific.

- [ ] **Step 4: Write the Remote SSH guide**

The guide must explain that saves modify server files directly and that the status bar must show `SSH: dmri-rocky`. Include a placeholder-only Mac `~/.ssh/config` example:

```sshconfig
Host dmri-rocky
    HostName SERVER_HOSTNAME
    User SERVER_USERNAME
    IdentityFile ~/.ssh/id_ed25519
```

Include remote-context verification:

```bash
ssh dmri-rocky
cat /etc/os-release
uname -m
pwd
```

Include environment and preflight:

```bash
export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh
./setup_rocky.sh --check
./run_pipeline.sh --validate-only config/subject.yaml
./run_pipeline.sh --dry-run config/subject.yaml
```

Explain how to run `"${CONDA_EXE}" run -n dmri-repro which python` in the remote terminal, then use `Python: Select Interpreter` in the Remote SSH window to select that returned server path. State that the interpreter selection is remote-workspace state and must not be committed as a machine-specific absolute path.

Include long-run handling without a pipe that could hide the pipeline exit code:

```bash
tmux new -s dmri-SUBJECT_ID
./run_pipeline.sh config/subject.yaml
# Detach with Ctrl-b, then d
tmux attach -t dmri-SUBJECT_ID
```

State explicitly that input paths resolve on the Rocky server, data is not copied to the Mac, SSHFS must not be used for bulk MRI computation, and VS Code disconnection does not end a process running inside `tmux`.

- [ ] **Step 5: Link the remote guide and run package tests**

Add `docs/REMOTE_VSCODE.md` to the README documentation list and to `REQUIRED_DOCUMENTS`. Then run:

```bash
python -m pytest tests/test_package_audit.py -q
```

Expected: all package, public-contract, JSON, and remote-workflow tests pass.

- [ ] **Step 6: Commit the remote operator workflow**

```bash
git add -- .vscode/extensions.json docs/REMOTE_VSCODE.md README.md \
  tests/test_package_audit.py
git commit -m "docs: add VS Code remote server workflow"
```

---

### Task 6: Cross-cutting verification and Rocky acceptance handoff

**Files:**
- Verify only; modify the owning task's files if a check fails.

**Interfaces:**
- Consumes: all deliverables from Tasks 1-5.
- Produces: a clean local test result plus a precise list of commands the operator can run in the connected Rocky VS Code terminal.

- [ ] **Step 1: Run shell and whitespace checks**

```bash
bash -n setup_rocky.sh run_pipeline.sh scripts/rocky_environment.sh \
  config/software.rocky.example.sh
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 2: Run the platform-focused regression set**

```bash
python -m pytest \
  tests/test_wrappers.py \
  tests/test_orchestrator.py \
  tests/test_noddi.py \
  tests/test_package_audit.py \
  tests/test_state.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the entire Python test suite**

```bash
python -m pytest -q
```

Expected: every test passes with no cache artifacts because project pytest configuration disables the cache provider.

- [ ] **Step 4: Scan supported public paths for stale macOS behavior**

```bash
rg -n 'setup_macos|INSTALL_MACOS|/Applications|hw\.memsize|Reproducible macOS|macOS package' \
  README.md pyproject.toml setup_rocky.sh run_pipeline.sh \
  src/dmri_pipeline tests docs/INSTALL_ROCKY.md docs/TROUBLESHOOTING.md \
  docs/REMOTE_VSCODE.md
```

Expected: no matches. Historical design and plan documents are intentionally excluded from this scan.

- [ ] **Step 5: Confirm repository state and commit any test-driven corrections**

```bash
git status --short
git log -6 --oneline
```

Expected: no uncommitted implementation files. If a preceding verification required a correction, amend the owning task with a new focused commit rather than creating an unrelated cleanup commit.

- [ ] **Step 6: Run real-server acceptance from the Remote SSH terminal**

These commands require the operator's real private paths and licences and therefore run only on the Rocky server:

```bash
cat /etc/os-release
uname -m
export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh
./setup_rocky.sh --check
./run_pipeline.sh --validate-only config/subject.yaml
./run_pipeline.sh --dry-run config/subject.yaml
```

Expected:

- `/etc/os-release` reports Rocky Linux 9.7;
- `uname -m` reports `x86_64`;
- setup reports exact FSL and MATLAB versions, `mexa64`, licensed Optimization Toolbox, a working MEX compile/load/run, pinned Python imports, atlas hashes, and disk space;
- validation exits 0 for valid server-side inputs;
- dry-run prints absolute server-side FSL and MATLAB commands without creating stage outputs.

Do not claim real-server acceptance until the operator supplies these command results.
