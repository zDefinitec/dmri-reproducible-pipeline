# Cluster Stage Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded TOPUP, EDDY, and NODDI scheduler jobs that reuse the existing pipeline and submit the next job only after the previous group succeeds.

**Architecture:** The Python orchestrator remains the only scientific execution engine and gains an optional named stage-group boundary. Shell adapters under `scripts/cluster/` validate private scheduler configuration, invoke the bounded pipeline, and use `CBIG_pbsubmit` to advance one immutable chain ID. Existing stage records remain the source of truth for prerequisite validation and resume.

**Tech Stack:** Python 3.11, pytest, Bash 4+, Rocky Linux 9.7, PBS through `CBIG_pbsubmit`, existing FSL/MATLAB pipeline.

## Global Constraints

- Do not copy or fork scientific stage implementations.
- Do not change TOPUP, BET, EDDY, DTI, DKI, NODDI, atlas, summary, or QC methods or thresholds.
- Preserve the existing ungrouped `run_pipeline.sh CONFIG.yaml` behavior.
- A nonzero, excluded, or review-held group must not submit its successor.
- A grouped run must reject missing or stale upstream evidence rather than silently executing it.
- A forced grouped run may name only a stage in that group and must invalidate that stage and every downstream stage.
- Do not use `sudo`, system `/tmp`, or delete existing result and `.invalidated` directories.
- Keep private paths and `config/cluster.local.sh` out of version control.
- Require an explicit positive `analysis.noddi_workers` for cluster-chain execution and `NODDI_NCPUS >= analysis.noddi_workers`.
- Remove the design and implementation-plan documents from the final tree after all implementation and verification work is complete, as requested by the user.

---

### Task 1: Bounded stage-group orchestration

**Files:**
- Modify: `src/dmri_pipeline/orchestrator.py`
- Modify: `src/dmri_pipeline/cli.py`
- Modify: `src/dmri_pipeline/__init__.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `STAGE_GROUPS: Mapping[str, tuple[str, ...]]` with public keys `topup`, `eddy`, and `noddi`; the ellipsis here is Python's variable-length tuple type syntax.
- Produces: `run_pipeline(config, mode, force_stage=None, stage_group=None) -> PipelineOutcome`.
- Produces: CLI option `--stage-group {topup,eddy,noddi}` and success status `GROUP_COMPLETE` mapped to exit code `0`.
- Consumes: existing `STAGE_ORDER`, `StageRunner.is_current`, stage invalidation, QC decision, subject lock, and stage specifications.

- [ ] **Step 1: Write failing group-boundary and grouped-execution tests**

Add literal expectations and behavior tests:

```python
EXPECTED_GROUPS = {
    "topup": EXPECTED_ORDER[0:5],
    "eddy": EXPECTED_ORDER[5:7],
    "noddi": EXPECTED_ORDER[7:15],
}

def test_stage_groups_are_exact_exhaustive_and_public() -> None:
    assert dict(STAGE_GROUPS) == EXPECTED_GROUPS
    assert tuple(stage for group in STAGE_GROUPS.values() for stage in group) == EXPECTED_ORDER
    assert dmri_pipeline.STAGE_GROUPS is STAGE_GROUPS

@pytest.mark.parametrize(
    ("group", "group_index"),
    (("topup", 0), ("eddy", 1), ("noddi", 2)),
)
def test_fake_group_run_executes_only_its_bounded_stages(
    subject_config,
    monkeypatch: pytest.MonkeyPatch,
    group: str,
    group_index: int,
) -> None:
    calls: list[str] = []
    _install_fake_pipeline(monkeypatch, subject_config, "INCLUDE", calls)
    group_order = ("topup", "eddy", "noddi")
    for prerequisite in group_order[:group_index]:
        run_pipeline(subject_config, "run", stage_group=prerequisite)
    calls.clear()
    outcome = run_pipeline(subject_config, "run", stage_group=group)
    assert outcome.status == "GROUP_COMPLETE"
    assert calls == list(EXPECTED_GROUPS[group])
```

Add separate tests proving EDDY and NODDI reject an absent or tampered upstream
record, a grouped run leaves downstream directories untouched, and a forced
stage outside its group is rejected before invalidation.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_orchestrator.py -k 'stage_group or grouped'
```

Expected: collection/import failure for missing `STAGE_GROUPS` or call failure
for the unsupported `stage_group` argument.

- [ ] **Step 3: Implement the minimal grouped orchestrator**

Add immutable literal group boundaries:

```python
STAGE_GROUPS = MappingProxyType(
    {
        "topup": STAGE_ORDER[0:5],
        "eddy": STAGE_ORDER[5:7],
        "noddi": STAGE_ORDER[7:15],
    }
)
```

Validate that the groups are exhaustive, ordered, and non-overlapping when the
module is loaded. Extend `run_pipeline` with `stage_group`. Use the existing
base-provenance runner for the first two QC-gate stages and the existing full
software-provenance runner for scientific stages. For stages before the group,
call `is_current` and read the stored QC decision without invoking the action.
For stages inside the group, retain `_reject_unsafe_existing_state` and
`StageRunner.run`. Never iterate stages after the group boundary.

Normal grouped runs must not invalidate anything. If `force_stage` is supplied,
verify membership in the selected group before calling the existing
`invalidate_from(STAGE_ORDER, force_stage)` cascade.

- [ ] **Step 4: Run focused orchestration tests and verify GREEN**

Run:

```bash
pytest -q tests/test_orchestrator.py -k 'stage_group or grouped'
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing CLI contract tests**

Add tests that exercise the real parser and assert:

```python
assert cli.main(["--stage-group", "topup", str(subject_config.config_path)]) == 0
assert captured_call == (subject_config, "run", None, "topup")
```

Also test an unknown group, `--validate-only --stage-group topup`, group-aware
`--force-stage` validation, `GROUP_COMPLETE -> 0`, and that the printed result
contains `group=topup`.

- [ ] **Step 6: Run CLI tests and verify RED**

Run:

```bash
pytest -q tests/test_cli.py -k 'group or outcome'
```

Expected: failures because the parser and call signature lack stage groups.

- [ ] **Step 7: Implement the CLI and public export**

Add `--stage-group` with choices from `STAGE_GROUPS`. Reject it with
`--validate-only`; allow it with normal runs and bounded dry-runs. Pass all
orchestrator arguments by keyword. Print `group=<name>` only for grouped
results. Export `STAGE_GROUPS` from `dmri_pipeline.__init__`.

- [ ] **Step 8: Add grouped dry-run tests, verify RED, then implement filtering**

Write a test that runs `dry-run` for EDDY against current prerequisites and
asserts its outcomes contain only `04_bet` and `05_eddy`, its command plan
contains BET/EDDY, contains no TOPUP/NODDI/JHU commands, and writes no files.

Refactor the dry-run command builder to associate each external command with
its owning stage before filtering. Keep the existing `_dry_run_commands`
return contract for ungrouped callers. Grouped dry-run may inspect upstream
state but must return only the selected group's outcomes.

Run:

```bash
pytest -q tests/test_orchestrator.py -k 'dry_run and group'
pytest -q tests/test_cli.py
```

Expected: all selected tests pass.

- [ ] **Step 9: Run all orchestration and CLI tests**

Run:

```bash
pytest -q tests/test_orchestrator.py tests/test_cli.py
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 1**

```bash
git add src/dmri_pipeline/orchestrator.py src/dmri_pipeline/cli.py src/dmri_pipeline/__init__.py tests/test_orchestrator.py tests/test_cli.py
git commit -m "feat: add bounded pipeline stage groups"
```

---

### Task 2: Validated cluster subject context

**Files:**
- Create: `src/dmri_pipeline/cluster.py`
- Modify: `src/dmri_pipeline/cli.py`
- Test: `tests/test_cluster.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `ClusterSubjectContext(subject_id: str, subject_output: Path, noddi_workers: int)`.
- Produces: `cluster_subject_context(config: PipelineConfig) -> ClusterSubjectContext`.
- Produces: CLI mode `--print-cluster-context`, which writes one sorted JSON object and exits zero without creating output directories or discovering FSL/MATLAB.

- [ ] **Step 1: Write failing context validation tests**

Cover a configured positive worker count and automatic selection:

```python
def test_cluster_context_uses_validated_subject_values(subject_config) -> None:
    context = cluster_subject_context(subject_config)
    assert context.subject_id == subject_config.subject_id
    assert context.subject_output == subject_config.subject_output
    assert context.noddi_workers == subject_config.analysis.noddi_workers

def test_cluster_context_rejects_automatic_noddi_workers(subject_config) -> None:
    automatic = replace(
        subject_config,
        analysis=replace(subject_config.analysis, noddi_workers="auto"),
    )
    with pytest.raises(ClusterConfigError, match="explicit"):
        cluster_subject_context(automatic)
```

- [ ] **Step 2: Run context tests and verify RED**

Run:

```bash
pytest -q tests/test_cluster.py
```

Expected: import failure because `dmri_pipeline.cluster` does not exist.

- [ ] **Step 3: Implement the immutable context helper**

The helper accepts only a validated `PipelineConfig`, rejects `"auto"`, and
returns paths without creating them. Its JSON representation contains exactly
`subject_id`, `subject_output`, and `noddi_workers`.

- [ ] **Step 4: Run context tests and verify GREEN**

Run:

```bash
pytest -q tests/test_cluster.py
```

Expected: all tests pass.

- [ ] **Step 5: Write and implement CLI context-mode tests via RED/GREEN**

Assert `--print-cluster-context CONFIG.yaml` prints one parseable JSON object,
does not call `run_pipeline`, and leaves `subject_output` absent. Reject its
combination with `--stage-group`, `--force-stage`, `--dry-run`, or
`--validate-only`. Map `ClusterConfigError` to exit code `2`.

Run after the failing test and again after implementation:

```bash
pytest -q tests/test_cli.py -k cluster_context
```

- [ ] **Step 6: Commit Task 2**

```bash
git add src/dmri_pipeline/cluster.py src/dmri_pipeline/cli.py tests/test_cluster.py tests/test_cli.py
git commit -m "feat: expose validated cluster run context"
```

---

### Task 3: Automatic `CBIG_pbsubmit` job chain

**Files:**
- Create: `scripts/cluster/common.sh`
- Create: `scripts/cluster/submit_subject_chain.sh`
- Create: `scripts/cluster/run_topup_subject.sh`
- Create: `scripts/cluster/run_eddy_subject.sh`
- Create: `scripts/cluster/run_noddi_subject.sh`
- Create: `config/cluster.example.sh`
- Modify: `.gitignore`
- Test: `tests/test_cluster_wrappers.py`

**Interfaces:**
- `submit_subject_chain.sh [--start-at topup|eddy|noddi] SUBJECT.yaml CLUSTER.local.sh`
- Each worker wrapper consumes `SUBJECT.yaml CLUSTER.local.sh CHAIN_ID`.
- Private configuration keys are the exact names specified in the design.
- Chain records and logs live below `CLUSTER_RUN_ROOT/CHAIN_ID`.

- [ ] **Step 1: Write failing launcher configuration tests**

Create a temporary package fixture containing the real cluster wrappers plus a
fake `run_pipeline.sh` and fake `CBIG_pbsubmit`. Run the launcher as a process,
not by grepping its source. Assert rejection of relative paths, absent files,
placeholder configuration, unsafe values, `noddi_workers="auto"`, and
`NODDI_NCPUS` below the explicit worker count.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
pytest -q tests/test_cluster_wrappers.py -k 'config or launcher'
```

Expected: failure because the cluster scripts do not exist.

- [ ] **Step 3: Implement shared validation and initial submission**

`common.sh` must use `set -euo pipefail`, validate every required key, reject
control characters, validate `HH:MM:SS`, memory strings such as `16G`, and
positive integer CPUs, and require absolute regular executable/config paths.
It obtains subject context only through:

```bash
DMRI_SOFTWARE_CONFIG="$DMRI_SOFTWARE_CONFIG" \
    "$REPO_ROOT/run_pipeline.sh" --print-cluster-context "$SUBJECT_CONFIG"
```

Parse the JSON with the Python interpreter from the configured `dmri-repro`
Conda environment and emit three tab-separated, control-character-free fields
for Bash `read`; do not parse YAML or JSON with shell text tools. Create the
chain directory below `CLUSTER_RUN_ROOT`, write the immutable inputs, and invoke
`CBIG_pbsubmit` with explicit `-cmd`, `-walltime`, `-mem`, `-ncpus`, `-name`,
`-jobout`, and `-joberr` values.

Build the `-cmd` value with a Bash `%q` helper over an argument array. Do not
use `eval`.

- [ ] **Step 4: Run launcher tests and verify GREEN**

Run:

```bash
pytest -q tests/test_cluster_wrappers.py -k 'config or launcher'
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing success-chain and stop-on-failure tests**

Exercise each real worker wrapper with fake bounded pipeline and submitter
processes. Assert the observable files and submitted argv:

- TOPUP exit `0` records completion and submits exactly one EDDY job;
- EDDY exit `0` records completion and submits exactly one NODDI job;
- NODDI exit `0` records terminal chain completion and submits nothing;
- exits `20`, `21`, `30`, `40`, and `50` are preserved and submit nothing;
- a failed successor submission records `submission_failed` and exits nonzero;
- re-entering the same successful worker with the same chain ID does not submit
  its successor twice.

- [ ] **Step 6: Run worker tests and verify RED**

Run:

```bash
pytest -q tests/test_cluster_wrappers.py -k 'success or failure or duplicate'
```

Expected: failures because the workers do not yet implement the chain.

- [ ] **Step 7: Implement the three worker wrappers**

Each wrapper loads the same immutable chain inputs, exports
`DMRI_SOFTWARE_CONFIG`, runs exactly one `--stage-group`, records the returned
status, and advances only on exact zero. Use an atomic lock directory and a
successor marker inside the chain directory around each submission. The NODDI
wrapper records `complete` and has no successor.

Do not remove lock/state files belonging to another process and do not delete
scientific outputs.

- [ ] **Step 8: Run all cluster-wrapper tests**

Run:

```bash
pytest -q tests/test_cluster_wrappers.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add .gitignore config/cluster.example.sh scripts/cluster tests/test_cluster_wrappers.py
git commit -m "feat: chain clustered diffusion jobs"
```

---

### Task 4: Distribution contract and operator documentation

**Files:**
- Modify: `src/dmri_pipeline/package_audit.py`
- Modify: `tests/test_package_audit.py`
- Modify: `tests/test_wrappers.py`
- Create: `docs/CLUSTER.md`
- Modify: `README.md`

**Interfaces:**
- The package audit accepts only the five named executable cluster wrappers in
  addition to the existing public executables; `common.sh` remains non-executable.
- `docs/CLUSTER.md` is the authoritative operator procedure.

- [ ] **Step 1: Write failing executable allowlist and wrapper tests**

Update the literal package export expectation to include:

```python
[
    "run_pipeline.sh",
    "scripts/cluster/run_eddy_subject.sh",
    "scripts/cluster/run_noddi_subject.sh",
    "scripts/cluster/run_topup_subject.sh",
    "scripts/cluster/submit_subject_chain.sh",
    "setup_rocky.sh",
]
```

Add wrapper relocation tests proving every executable derives the repository
root from its own physical location and works when invoked from another
directory.

- [ ] **Step 2: Run distribution tests and verify RED**

Run:

```bash
pytest -q tests/test_package_audit.py tests/test_wrappers.py -k 'cluster or exported_package'
```

Expected: failure because the allowlist and documented wrapper contract have
not been updated.

- [ ] **Step 3: Update the allowlist and documentation**

Add only the four executable worker/launcher relative paths to the package
allowlist. Write `docs/CLUSTER.md` with:

- headnode submission prerequisites;
- private configuration creation;
- explicit `analysis.noddi_workers` requirement;
- full-chain and `--start-at` examples;
- resource tuning from completed PBS `.STATS` files;
- log and chain-state locations;
- exit code meanings;
- recovery without deleting outputs; and
- the limitation that `CBIG_pbsubmit` returning zero does not prove the queued
  job later ran successfully.

Link the document from `README.md`. Do not include real usernames, hosts,
subject identifiers, or private absolute paths.

- [ ] **Step 4: Run focused distribution tests and verify GREEN**

Run:

```bash
pytest -q tests/test_package_audit.py tests/test_wrappers.py -k 'cluster or exported_package'
```

Expected: all selected tests pass.

- [ ] **Step 5: Run shell syntax and formatting checks**

Run:

```bash
for script in scripts/cluster/*.sh; do bash -n "$script"; done
git diff --check
```

Expected: both commands exit zero with no diagnostics.

- [ ] **Step 6: Commit Task 4**

```bash
git add README.md docs/CLUSTER.md src/dmri_pipeline/package_audit.py tests/test_package_audit.py tests/test_wrappers.py
git commit -m "docs: add clustered pipeline workflow"
```

---

### Task 5: Final verification and removal of process documents

**Files:**
- Delete: `docs/superpowers/specs/2026-09-03-cluster-stage-chain-design.md`
- Delete: `docs/superpowers/plans/2026-09-03-cluster-stage-chain.md`

**Interfaces:**
- Produces: a clean final source tree containing only product code, product tests,
  example configuration, and operator documentation for this feature.

- [ ] **Step 1: Run all focused feature tests**

```bash
pytest -q tests/test_orchestrator.py tests/test_cli.py tests/test_cluster.py tests/test_cluster_wrappers.py tests/test_package_audit.py tests/test_wrappers.py
```

Expected: all tests pass.

- [ ] **Step 2: Run the complete test suite**

```bash
pytest -q
```

Expected: all tests pass, with only the repository's documented MATLAB-related
skips.

- [ ] **Step 3: Audit the final behavior against the design**

Confirm from tests and source that group boundaries are exact, ungrouped runs
remain compatible, only zero advances the chain, private configuration is
ignored, no scientific threshold changed, and no deletion targets scientific
outputs.

- [ ] **Step 4: Remove process-only documents**

Use the patch tool to delete the two design/plan files listed above. Do not
delete `docs/CLUSTER.md` or any product documentation.

- [ ] **Step 5: Re-run final verification after cleanup**

```bash
pytest -q
git diff --check
git status --short
```

Expected: the test suite passes, diff checks are clean, and status shows only
the intentional process-document deletions before the final commit.

- [ ] **Step 6: Commit cleanup**

```bash
git add docs/superpowers/specs/2026-09-03-cluster-stage-chain-design.md docs/superpowers/plans/2026-09-03-cluster-stage-chain.md
git commit -m "chore: remove cluster design artifacts"
```
