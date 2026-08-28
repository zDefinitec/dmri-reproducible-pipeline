# Split EDDY Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator prepare a subject through BET, run only EDDY for one subject or a sequential cohort, and then resume the complete Rocky-server pipeline without weakening validation, provenance, QC gates, or atomic stage state.

**Architecture:** Add an immutable stage-selection value that derives one execution range and, for single-stage mode, an exact-current upstream-validation range. Refactor runtime provenance from one global FSL-plus-MATLAB snapshot into deterministic per-stage snapshots so bounded EDDY discovers FSL but not MATLAB while ordinary full runs retain strict dependencies. Keep EDDY inside the existing `StageRunner`, add validated timing evidence to its promoted outputs and final report, and make the cohort shell script a thin sequential caller of the normal Rocky entry point.

**Tech Stack:** Python 3.11, argparse, dataclasses, pytest, Bash, FSL 6.0.7.18 CPU EDDY/EDDY QUAD, MATLAB R2025a for downstream NODDI only, Rocky Linux 9.7 x86_64, VS Code Remote SSH, tmux.

## Global Constraints

- All MRI computation, inputs, logs, work directories, and outputs remain on the Rocky Linux 9.7 x86_64 server. The Mac is only an SSH/VS Code client.
- The ordinary `./run_pipeline.sh CONFIG.yaml` command and its complete ordered result remain supported.
- `--stop-after STAGE` executes the ordered prefix ending at `STAGE` and succeeds with `PARTIAL_COMPLETE`.
- `--only-stage STAGE` proves every upstream stage exact-current, enforces the pre-denoise QC gate, executes or skips exactly `STAGE`, and succeeds with `STAGE_COMPLETE`.
- `--stop-after` and `--only-stage` are mutually exclusive. Neither combines with `--validate-only` or `--dry-run`.
- `--force-stage` is legal only inside the selected execution range. In `--only-stage` mode it must equal the selected stage.
- A forced stage still archives that stage and every downstream stage before execution; bounded execution does not preserve stale downstream results.
- Only the software needed to validate and execute the bounded range is discovered. In particular, EDDY-only work requires FSL but must not discover MATLAB.
- Normal full execution retains the current fail-closed FSL and MATLAB dependency checks.
- EDDY keeps the current full PA DWI input and current production flags; add only explicit `--fwhm=0` and `--flm=quadratic` defaults.
- EDDY timing is observational evidence, never a pass/fail threshold or a two-hour promise.
- The cohort wrapper is sequential, accepts explicit YAML arguments, continues after participant failures, and returns nonzero if any participant fails.
- Do not introduce GPU EDDY, scheduler submission, automatic subject discovery, concurrent EDDY jobs, or scientific model changes.
- `orchestrator.py` is already source evidence for every stage. Deploying this release makes older stage records non-current by design; never replace the code under a subject that is actively running.

## File Map

- `src/dmri_pipeline/cli.py`: bounded CLI flags, cross-option validation, status-to-exit mapping.
- `src/dmri_pipeline/eddy_timing.py`: one strict timing schema shared by stage validation and reports.
- `src/dmri_pipeline/orchestrator.py`: `StageSelection`, bounded execution, upstream proof, stage-scoped dependency/provenance, EDDY timing validation and paths.
- `src/dmri_pipeline/fsl.py`: explicit EDDY smoothing and field-model arguments.
- `src/dmri_pipeline/report.py`: timing input validation plus JSON, Markdown, and PDF presentation.
- `src/dmri_pipeline/__init__.py`: public `StageSelection` export.
- `src/dmri_pipeline/package_audit.py`: allow the new executable wrapper.
- `run_eddy_batch.sh`: sequential EDDY-only cohort wrapper.
- `tests/test_cli.py`: parser conflicts, forwarding, and success exit codes.
- `tests/test_orchestrator.py`: range semantics, upstream proof, dependency discovery, forced invalidation, and timing validation.
- `tests/test_fsl.py`: exact EDDY command wiring.
- `tests/test_report.py`: timing propagation and deterministic report rendering.
- `tests/test_wrappers.py`: sequential continuation and final batch status.
- `tests/test_package_audit.py`, `tests/test_state.py`: executable inventory and release metadata.
- `README.md`, `docs/PIPELINE.md`, `docs/OUTPUTS.md`, `docs/TROUBLESHOOTING.md`, `docs/REMOTE_VSCODE.md`: operator contract and migration warning.
- `VERSION`, `pyproject.toml`: public minor-version bump to `2.1.0`.

---

### Task 1: Define and parse bounded stage selection

**Files:**
- Modify: `src/dmri_pipeline/orchestrator.py`
- Modify: `src/dmri_pipeline/cli.py`
- Modify: `src/dmri_pipeline/__init__.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: public immutable `StageSelection(stop_after=None, only_stage=None)`; CLI options `--stop-after STAGE` and `--only-stage STAGE`; success statuses `PARTIAL_COMPLETE` and `STAGE_COMPLETE`, both mapped to exit code `0`.
- Preserves: positional single YAML configuration, current nonmutating modes, current exception-to-exit mapping, and current `--force-stage` spelling.

- [ ] **Step 1: Add failing CLI tests for accepted selections and forwarding**

Update the monkeypatched pipeline callable to accept a keyword-only selection and capture it:

```python
captured: dict[str, object] = {}

def fake_run_pipeline(config, mode, force_stage=None, *, selection=None):
    captured.update(
        config=config,
        mode=mode,
        force_stage=force_stage,
        selection=selection,
    )
    return PipelineOutcome(config.subject_id, "STAGE_COMPLETE", (), config.subject_output)

monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
assert cli.main(["--only-stage", "05_eddy", str(config_path)]) == 0
assert captured["selection"] == StageSelection(only_stage="05_eddy")
```

Add the matching `--stop-after 04_bet` test and assert `PARTIAL_COMPLETE` also exits `0`.

- [ ] **Step 2: Add failing CLI tests for every illegal option combination**

Parametrize these argument vectors and assert exit code `2`, an `ERROR:` line on stderr, and no pipeline call:

```python
(
    ["--stop-after", "04_bet", "--only-stage", "05_eddy", config],
    ["--validate-only", "--stop-after", "04_bet", config],
    ["--dry-run", "--only-stage", "05_eddy", config],
    ["--only-stage", "05_eddy", "--force-stage", "04_bet", config],
    ["--stop-after", "04_bet", "--force-stage", "05_eddy", config],
)
```

Also prove the intended rerun is accepted:

```python
assert cli.main(
    ["--only-stage", "05_eddy", "--force-stage", "05_eddy", str(config_path)]
) == 0
```

- [ ] **Step 3: Run the focused CLI tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: failures mention unrecognized bounded flags, the old `run_pipeline` call signature, or unknown success statuses.

- [ ] **Step 4: Implement `StageSelection` and central validation**

Add the immutable value beside `PipelineOutcome` in `orchestrator.py`:

```python
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
```

Add central Python-API validation:

```python
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
```

Before loading the YAML, make the CLI perform the corresponding checks and raise `_CLIError`; this preserves usage-error exit code `2`. Call `_validate_selection` again inside `run_pipeline` so Python callers cannot bypass the contract.

Extend the Python entry point now, while leaving its existing complete traversal in place until Task 3:

```python
def run_pipeline(
    config: PipelineConfig,
    mode: str,
    force_stage: str | None = None,
    *,
    selection: StageSelection | None = None,
) -> PipelineOutcome:
    selection = selection or StageSelection()
    _validate_selection(mode, force_stage, selection)
    # Existing complete-run body; Task 3 applies the calculated range.
```

- [ ] **Step 5: Parse the flags and forward one explicit selection**

Place the new flags in one mutually exclusive group and keep nonmutating modes in their existing group:

```python
bounds = parser.add_mutually_exclusive_group()
bounds.add_argument("--stop-after", choices=STAGE_ORDER)
bounds.add_argument("--only-stage", choices=STAGE_ORDER)
```

After parsing, reject a bound combined with a nonmutating mode. Construct and forward:

```python
selection = StageSelection(
    stop_after=namespace.stop_after,
    only_stage=namespace.only_stage,
)
outcome = run_pipeline(
    config,
    mode,
    namespace.force_stage,
    selection=selection,
)
```

Add `PARTIAL_COMPLETE` and `STAGE_COMPLETE` to the zero-exit status mapping. Export `StageSelection` from `src/dmri_pipeline/__init__.py`.

- [ ] **Step 6: Run the CLI tests and confirm they pass**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: all CLI tests pass; invalid combinations return `2` without loading external software.

- [ ] **Step 7: Commit the public CLI contract**

```bash
git add src/dmri_pipeline/cli.py src/dmri_pipeline/orchestrator.py src/dmri_pipeline/__init__.py tests/test_cli.py
git commit -m "feat: add bounded stage selection CLI"
```

---

### Task 2: Make software discovery and provenance stage-scoped

**Files:**
- Modify: `src/dmri_pipeline/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `_stage_software_provenance(runtime, stage_name)` and a fresh `StageRunner` context for the specific stage being checked or executed.
- Preserves: exact software evidence in `.stage_complete.json`, full FSL-plus-MATLAB discovery for ordinary complete runs and nonmutating full-plan validation.

- [ ] **Step 1: Add failing unit tests for stage dependency evidence**

Instrument fake FSL/MATLAB discovery with ordered call lists and call the provenance helper directly. Assert:

```python
assert _stage_software_provenance(runtime, "00_input_audit") == BASE
assert discoveries == []

assert _stage_software_provenance(runtime, "05_eddy") == BASE | FSL
assert discoveries == ["fsl"]

runtime, discoveries = fresh_runtime_and_trace()
assert _stage_software_provenance(runtime, "08_noddi") == BASE | MATLAB
assert discoveries == ["matlab"]
```

Also test an unknown stage and assert it raises `ValueError` without discovery.

- [ ] **Step 2: Add a failing ordinary-run record identity test**

Run the existing unbounded fake pipeline and inspect the promoted records. Assert a base-only stage has neither external package, EDDY has FSL evidence but no MATLAB evidence, and NODDI has MATLAB evidence but no FSL evidence:

```python
assert record_software("00_input_audit") == BASE
assert record_software("05_eddy") == BASE | FSL
assert record_software("08_noddi") == BASE | MATLAB
```

The ordinary run must still call both discovery functions before its scientific traversal, preserving the current strict full-run dependency preflight.

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_orchestrator.py -q -k 'software or matlab or bounded_provenance'
```

Expected: the helper is absent and every scientific stage currently records the same global FSL-plus-MATLAB mapping.

- [ ] **Step 4: Split deterministic software evidence by stage**

Define the external dependency sets once:

```python
_FSL_STAGES = frozenset(
    {"01_denoise", "03_topup", "04_bet", "05_eddy", "09_jhu_48roi"}
)
_MATLAB_STAGES = frozenset({"08_noddi"})


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
```

Extract `_fsl_software_provenance(installation)` and `_matlab_software_provenance(installation)` from the existing `_software_provenance`. Keep `_software_provenance(runtime)` as their deterministic union for the existing complete dry-run/validation contract.

Do not use a current process environment dump, licence status, wall-clock timestamp, or executable modification time as evidence; records must be reproducible and equality-comparable.

- [ ] **Step 5: Build one runner per stage and remove accidental discovery from QC/report contexts**

Add:

```python
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
```

Change `_qc_context` and `_report_context` to receive the already-calculated stage software mapping. They must not call `_software_provenance(runtime)` themselves. Rebuild a stage spec immediately before checking it only when its lazy FSL/MATLAB installation is now available, so command construction and its stage provenance use the same discovered installation.

For the unbounded body in this task, retain the explicit `runtime.require_fsl()` and `runtime.require_matlab()` preflight, then iterate with one stage-scoped runner per spec. Task 3 will make that preflight conditional on an unbounded selection.

- [ ] **Step 6: Run provenance tests and the complete orchestrator module**

Run:

```bash
python -m pytest tests/test_orchestrator.py -q
```

Expected: all tests pass; ordinary full execution still requires both installations, but every completion record contains only the deterministic software evidence relevant to its own stage.

- [ ] **Step 7: Commit the dependency-boundary refactor**

```bash
git add src/dmri_pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "refactor: scope software provenance by stage"
```

---

### Task 3: Execute bounded ranges without bypassing state or QC gates

**Files:**
- Modify: `src/dmri_pipeline/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `StageSelection` plus optional valid `force_stage`.
- Produces: exact ordered outcomes for the execution range; upstream validation before an `only_stage`; boundary success statuses.
- Preserves: subject locking, unsafe-state rejection, manual review transition, atomic promotion, force archival, and existing EXCLUDE/HOLD outcomes.

- [ ] **Step 1: Add failing prefix-range tests**

Use the fake plan and call:

```python
outcome = run_pipeline(
    config,
    "run",
    selection=StageSelection(stop_after="04_bet"),
)
assert outcome.status == "PARTIAL_COMPLETE"
assert tuple(stage.stage for stage in outcome.stages) == STAGE_ORDER[:6]
assert not (config.subject_output / "05_eddy").exists()
```

Add a prefix ending at `00_input_audit` to prove the QC decision is not read before its stage exists. Add EXCLUDE and HOLD fixtures for a prefix that reaches `00_pre_denoise_motion_qc`; assert their existing statuses and exit semantics win over `PARTIAL_COMPLETE`.

- [ ] **Step 2: Add failing single-stage upstream-proof tests**

Cover all four refusal classes before the selected action runs:

```python
@pytest.mark.parametrize(
    "upstream_state",
    ("missing", "stale_input", "EXCLUDE", "HOLD_FOR_REVIEW"),
)
def test_only_eddy_refuses_invalid_upstream(...):
    action_called = False
    with pytest.raises(StageStateError):
        run_pipeline(
            config,
            "run",
            selection=StageSelection(only_stage="05_eddy"),
        )
    assert action_called is False
```

For missing/stale stage records, assert the error names the first non-current upstream stage. For QC decisions, preserve `EXCLUDED`/`HOLD_FOR_REVIEW` as `PipelineOutcome` results when the current QC record itself carries those decisions; do not launch EDDY.

- [ ] **Step 3: Add failing exact-current, forced-rerun, and downstream-archive tests**

Seed through EDDY, then prove:

```python
already_current = run_pipeline(
    config,
    "run",
    selection=StageSelection(only_stage="05_eddy"),
)
assert already_current.status == "STAGE_COMPLETE"
assert already_current.stage_statuses == (("05_eddy", "SKIPPED"),)
```

Create exact-current fake `06_dti` and `07_dki` outputs, then invoke `force_stage="05_eddy"` with the single-stage selection. Assert EDDY executes, its former final is archived, downstream finals are archived, and no downstream action executes.

In the same test group, instrument discovery and assert the bounded dependency contract:

```python
assert discoveries_for(StageSelection(stop_after="00_input_audit")) == []
assert discoveries_for(StageSelection(stop_after="04_bet")) == ["fsl"]
assert discoveries_for(StageSelection(only_stage="05_eddy")) == ["fsl"]
assert discoveries_for(StageSelection(only_stage="08_noddi")) == ["fsl", "matlab"]
assert discoveries_for(StageSelection()) == ["fsl", "matlab"]
```

For `only_stage`, seed exact-current upstream records with the stage-scoped evidence from Task 2. Replace `discover_matlab` with a raising stub in the EDDY-only case and prove it is never called.

Finally, run a prefix through `05_eddy`, then resume with the ordinary full selection using the same fake installations. Assert the prefix stages report `SKIPPED`, proving bounded and ordinary commands calculate identical stage records:

```python
run_pipeline(
    config,
    "run",
    selection=StageSelection(stop_after="05_eddy"),
)
resumed = run_pipeline(config, "run", selection=StageSelection())
assert dict(resumed.stage_statuses)["05_eddy"] == "SKIPPED"
```

- [ ] **Step 4: Run bounded-execution tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_orchestrator.py -q -k 'stop_after or only_stage or force_bounded'
```

Expected: current runner traverses the full plan and cannot distinguish upstream proof from selected execution.

- [ ] **Step 5: Validate only the selected plan sources and required resources**

At the beginning of `run_pipeline`, normalize `selection = selection or StageSelection()` and validate mode/force rules. For a normal full run keep the current complete validation. For a bound:

```python
required_names = selection.required_names
required_plan = [spec for spec in plan if spec.name in required_names]
_validate_plan_sources(required_plan)
if "09_jhu_48roi" in required_names:
    validate_jhu_resource(_ATLAS_IMAGE, _ATLAS_XML)
```

Do not let a downstream JHU resource or MATLAB installation block `04_bet` or `05_eddy`. Continue performing the read-only input audit before any subject mutation.

- [ ] **Step 6: Add exact-current upstream verification**

Inside the subject lock, before force invalidation or selected execution, verify every name before `only_stage`:

```python
def _require_current_upstream(
    config: PipelineConfig,
    runtime: _Runtime,
    plan: Sequence[StageSpec],
    selected_stage: str,
) -> PipelineOutcome | None:
    for spec in plan[: STAGE_ORDER.index(selected_stage)]:
        runner = _stage_runner(config, runtime, spec.name)
        _reject_unsafe_existing_state(runner, (spec,))
        if not runner.is_current(spec):
            raise StageStateError(
                f"upstream stage is not exact-current: {spec.name}"
            )
        if spec.name == "00_pre_denoise_motion_qc":
            decision = _read_qc_decision(runner.final_dir(spec.name))
            if decision.status in {"EXCLUDE", "HOLD_FOR_REVIEW"}:
                return PipelineOutcome(
                    config.subject_id,
                    decision.status,
                    (),
                    config.subject_output,
                )
            if decision.status not in _CONTINUE_QC:
                raise StageStateError(
                    f"unsupported stripe-QC decision: {decision.status}"
                )
    return None
```

The concrete implementation may return the QC outcome through a small helper result rather than this exact signature, but it must check `StageRunner.is_current(spec)` with the exact stage-scoped context and must stop before the selected external command.

- [ ] **Step 7: Execute exactly the requested range under the existing lock**

Use `selection.execution_names` to select specs. Before execution:

1. Apply the existing safe manual-review transition only when the selection will execute the input/QC gate in normal ordered-prefix mode.
2. If forcing, call `invalidate_from(STAGE_ORDER, force_stage)` once under the subject lock, using a safe runner whose subject root is identical.
3. For each selected spec, build the stage-scoped runner, call `_reject_unsafe_existing_state`, then `runner.run(spec)`.
4. Read the QC decision immediately after executing or skipping `00_pre_denoise_motion_qc` when it is in the execution range.
5. Return `selection.success_status` only after the boundary stage succeeds or skips.

Do not pre-create a selected final directory, copy upstream outputs, synthesize records, or treat directory existence as current state.

- [ ] **Step 8: Prove the ordinary complete path is unchanged**

Add or retain assertions that a normal run:

```python
outcome = run_pipeline(config, "run")
assert outcome.status == "COMPLETE"
assert tuple(stage.stage for stage in outcome.stages) == STAGE_ORDER
```

Run:

```bash
python -m pytest tests/test_orchestrator.py tests/test_cli.py -q
```

Expected: all bounded, force, QC, locking, and legacy complete-run tests pass.

- [ ] **Step 9: Commit bounded orchestration**

```bash
git add src/dmri_pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: execute validated bounded stage ranges"
```

---

### Task 4: Make the EDDY command explicit and record validated runtime evidence

**Files:**
- Create: `src/dmri_pipeline/eddy_timing.py`
- Modify: `src/dmri_pipeline/fsl.py`
- Modify: `src/dmri_pipeline/orchestrator.py`
- Modify: `tests/test_fsl.py`
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: EDDY arguments `--fwhm=0` and `--flm=quadratic`; promoted `05_eddy/eddy_timing.json`.
- Preserves: full Gibbs-corrected 4D DWI `--imain`, hifi brain mask, EDDY index/acquisition parameters, canonical gradients, TOPUP basename, outlier replacement, CNR maps, residuals, shelled declaration, threads, and EDDY QUAD.

- [ ] **Step 1: Tighten the exact EDDY command test**

Extend `test_eddy_has_exact_acquisition_gradient_mask_and_thread_wiring` so the expected command contains these exact tokens once:

```python
"--fwhm=0",
"--flm=quadratic",
```

Retain assertions that `--imain` is the full `02_gibbs/dMRI_PA_pca_gibbs.nii.gz`, not a corrected-b0 image or an EDDY output, and that `--repol`, `--cnr_maps`, `--residuals`, `--data_is_shelled`, `--nthr=...`, and `--out=...` remain present.

- [ ] **Step 2: Add failing timing-schema and output-validation tests**

Use one exact schema:

```json
{
  "schema_version": 1,
  "eddy_command_seconds": 7200.0,
  "eddy_quad_seconds": 240.0,
  "stage_action_seconds": 7440.5,
  "eddy_command_includes_cnr_maps": true,
  "eddy_command_includes_residuals": true
}
```

Add tests that reject missing/extra keys, booleans in numeric fields, NaN/infinity, negative values, false inclusion flags, and totals smaller than the two component durations. Update every existing valid EDDY fixture helper, including `_populate_valid_eddy`, to write a valid timing file.

- [ ] **Step 3: Add a failing deterministic monotonic-timing test**

Monkeypatch `time.monotonic` in `orchestrator.py` to a known iterator and stub both external calls. Assert the file records the complete EDDY call separately from EDDY QUAD and includes post-QUAD sanitization in the stage total:

```python
ticks = iter((10.0, 10.0, 130.0, 130.0, 145.0, 146.0))
monkeypatch.setattr(orchestrator.time, "monotonic", lambda: next(ticks))
assert payload["eddy_command_seconds"] == 120.0
assert payload["eddy_quad_seconds"] == 15.0
assert payload["stage_action_seconds"] == 136.0
```

The exact number of sampled ticks must match the final action implementation; do not time by filesystem timestamps or CPU time.

- [ ] **Step 4: Run focused tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_fsl.py tests/test_orchestrator.py -q -k 'eddy'
```

Expected: the two explicit command tokens and `eddy_timing.json` are absent.

- [ ] **Step 5: Add the explicit command defaults**

In `build_eddy_command`, add:

```python
"--fwhm=0",
"--flm=quadratic",
```

Keep the discovered executable name. Do not hard-code `eddy_openmp`: the accepted Rocky FSL installation may expose `eddy`, `eddy_cpu`, or `eddy_openmp`, and discovery already records the validated path.

- [ ] **Step 6: Implement one shared strict timing schema**

Create `src/dmri_pipeline/eddy_timing.py` with an immutable value and exact JSON conversion:

```python
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_KEYS = frozenset(
    {
        "schema_version",
        "eddy_command_seconds",
        "eddy_quad_seconds",
        "stage_action_seconds",
        "eddy_command_includes_cnr_maps",
        "eddy_command_includes_residuals",
    }
)


class EddyTimingError(ValueError):
    """EDDY timing JSON is missing, unreadable, or scientifically ambiguous."""


@dataclass(frozen=True)
class EddyTiming:
    eddy_command_seconds: float
    eddy_quad_seconds: float
    stage_action_seconds: float


def parse_eddy_timing(payload: object) -> EddyTiming:
    if not isinstance(payload, Mapping) or set(payload) != _KEYS:
        raise EddyTimingError("EDDY timing must contain the exact version-1 key set")
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise EddyTimingError("EDDY timing schema_version must be integer 1")
    for key in (
        "eddy_command_includes_cnr_maps",
        "eddy_command_includes_residuals",
    ):
        if payload[key] is not True:
            raise EddyTimingError(f"EDDY timing {key} must be true")

    values: dict[str, float] = {}
    for key in (
        "eddy_command_seconds",
        "eddy_quad_seconds",
        "stage_action_seconds",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EddyTimingError(f"EDDY timing {key} must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise EddyTimingError(f"EDDY timing {key} must be finite and nonnegative")
        values[key] = normalized

    timing = EddyTiming(**values)
    component_total = timing.eddy_command_seconds + timing.eddy_quad_seconds
    if timing.stage_action_seconds < component_total:
        raise EddyTimingError("EDDY stage total cannot be smaller than its commands")
    return timing


def read_eddy_timing(path: Path) -> EddyTiming:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EddyTimingError("cannot read EDDY timing JSON") from error
    return parse_eddy_timing(payload)


def write_eddy_timing(path: Path, timing: EddyTiming) -> None:
    if not isinstance(timing, EddyTiming):
        raise TypeError("timing must be EddyTiming")
    payload = {
        "schema_version": 1,
        "eddy_command_seconds": timing.eddy_command_seconds,
        "eddy_quad_seconds": timing.eddy_quad_seconds,
        "stage_action_seconds": timing.stage_action_seconds,
        "eddy_command_includes_cnr_maps": True,
        "eddy_command_includes_residuals": True,
    }
    parse_eddy_timing(payload)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

Use one module-level exact key tuple and raise `EddyTimingError(ValueError)` for malformed JSON, I/O errors, wrong keys/types, nonfinite/negative values, false inclusion flags, or an impossible total. This separate module avoids the circular dependency that would result if `report.py` imported a private parser from `orchestrator.py`.

- [ ] **Step 7: Time the complete stage action with a monotonic clock**

Add `import time` and write the timing file only after both external commands and QUAD sanitization succeed:

```python
stage_started = time.monotonic()
eddy_started = time.monotonic()
run_fsl_command(eddy_command, cwd=work)
eddy_finished = time.monotonic()

quad_started = time.monotonic()
run_fsl_command(eddy_quad_command, cwd=work)
quad_finished = time.monotonic()
_write_sanitized_eddy_quad(...)
stage_finished = time.monotonic()

write_eddy_timing(
    work / "eddy_timing.json",
    EddyTiming(
        eddy_command_seconds=eddy_finished - eddy_started,
        eddy_quad_seconds=quad_finished - quad_started,
        stage_action_seconds=stage_finished - stage_started,
    ),
)
```

If either command or sanitization fails, no final stage is promoted; the work directory remains subject to the existing non-resumable-work safety rule.

- [ ] **Step 8: Validate timing as a required EDDY output**

Add `eddy_timing` to `_paths`, the EDDY validator required set, report input paths, and report context. Call the shared `read_eddy_timing` from both EDDY validation and report assembly. It must:

- require the exact key set above;
- require integer `schema_version == 1` without accepting `True`;
- require true booleans for both inclusion fields;
- require real, finite, nonnegative duration values without accepting booleans;
- require `stage_action_seconds >= eddy_command_seconds + eddy_quad_seconds`.

- [ ] **Step 9: Run EDDY tests and commit**

Run:

```bash
python -m pytest tests/test_fsl.py tests/test_orchestrator.py -q -k 'eddy'
```

Expected: all focused EDDY command, timing, promotion, and validation tests pass.

Commit:

```bash
git add src/dmri_pipeline/eddy_timing.py src/dmri_pipeline/fsl.py src/dmri_pipeline/orchestrator.py tests/test_fsl.py tests/test_orchestrator.py
git commit -m "feat: record explicit EDDY runtime evidence"
```

---

### Task 5: Propagate EDDY timing into deterministic reports

**Files:**
- Modify: `src/dmri_pipeline/eddy_timing.py`
- Modify: `src/dmri_pipeline/report.py`
- Modify: `src/dmri_pipeline/orchestrator.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: exact validated `ReportContext.eddy_timing_json`.
- Produces: nested EDDY timing facts in `report.json`, explicit timing rows in Markdown, and the same three values on the EDDY technical-QC PDF page.
- Preserves: the top-level `REPORT_JSON_KEYS` order and all existing report output names.

- [ ] **Step 1: Add a valid timing fixture to every report case**

Write the exact schema in `make_report_case` and pass it into the context:

```python
eddy_timing = eddy_dir / "eddy_timing.json"
eddy_timing.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "eddy_command_seconds": 7200.0,
            "eddy_quad_seconds": 240.0,
            "stage_action_seconds": 7441.0,
            "eddy_command_includes_cnr_maps": True,
            "eddy_command_includes_residuals": True,
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
```

- [ ] **Step 2: Add failing JSON, Markdown, PDF, and immutability tests**

Assert the nested JSON is exact:

```python
assert payload["eddy"]["runtime_seconds"] == {
    "eddy_command_including_cnr_and_residuals": 7200.0,
    "eddy_quad": 240.0,
    "stage_action_total": 7441.0,
}
```

Assert Markdown contains all three labels and values, including the phrase `includes CNR maps and residuals`. Build the guarded `_ReportData` as the existing PDF summary tests do, join `_pdf_page4_summary(data.payload)`, and assert it contains `EDDY command`, `EDDY QUAD`, and `stage action total`. Add a mutation-during-report test for `eddy_timing.json`, matching the existing immutable-input guard tests.

- [ ] **Step 3: Run report tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_report.py -q
```

Expected: `ReportContext` does not yet accept the timing path and rendered reports lack timing facts.

- [ ] **Step 4: Add the timing path to the guarded report inputs**

Add `eddy_timing_json: Path` to `ReportContext`. Include it in `_validate_path_assignment` and `_context_inputs` so it is snapshotted and checked like parameters, motion, outlier, and QUAD data. Read it through `eddy_timing.read_eddy_timing`; catch `EddyTimingError` and raise `ReportError` with the original exception as its cause. Do not duplicate a looser report-only validator.

- [ ] **Step 5: Add timing to `_eddy_facts` without changing top-level keys**

Append this ordered entry inside the existing EDDY mapping:

```python
(
    "runtime_seconds",
    OrderedDict(
        (
            (
                "eddy_command_including_cnr_and_residuals",
                timing["eddy_command_seconds"],
            ),
            ("eddy_quad", timing["eddy_quad_seconds"]),
            ("stage_action_total", timing["stage_action_seconds"]),
        )
    ),
),
```

Leave `REPORT_JSON_KEYS` unchanged because `runtime_seconds` is nested under the existing `eddy` key.

- [ ] **Step 6: Render clear Markdown and PDF labels**

Add three rows to the EDDY technical QC Markdown table:

```text
EDDY command (includes CNR maps and residuals) | seconds | 7200
EDDY QUAD | seconds | 240
05_eddy stage action total | seconds | 7441
```

Add the same values to the existing PDF page that summarizes EDDY motion/outliers. Keep values in seconds so the report preserves measurement precision; documentation may provide a human-readable conversion example but must not replace raw values.

- [ ] **Step 7: Run deterministic report tests and commit**

Run:

```bash
python -m pytest tests/test_report.py -q
```

Expected: all JSON, Markdown, PDF, symlink, mutation, and deterministic-output tests pass.

Commit:

```bash
git add src/dmri_pipeline/eddy_timing.py src/dmri_pipeline/report.py src/dmri_pipeline/orchestrator.py tests/test_report.py
git commit -m "feat: report EDDY runtime evidence"
```

---

### Task 6: Add the sequential cohort EDDY wrapper

**Files:**
- Create: `run_eddy_batch.sh`
- Modify: `src/dmri_pipeline/package_audit.py`
- Modify: `tests/test_wrappers.py`
- Modify: `tests/test_package_audit.py`

**Interfaces:**
- Consumes: one or more explicit YAML paths as positional arguments.
- Calls: `run_pipeline.sh --only-stage 05_eddy CONFIG.yaml` exactly once per supplied argument, in argument order.
- Produces: per-config start/result lines, final failure summary, exit `0` only when all calls return `0`, otherwise exit `1`.

- [ ] **Step 1: Add failing wrapper usage and argument-forwarding tests**

Follow the current sourceable-wrapper test seam. The public script should expose `_dmri_run_eddy_batch_main RUNNER CONFIG...` and invoke it with its sibling `run_pipeline.sh` only when executed directly. Test:

```bash
source "$DMRI_TEST_WRAPPER"
_dmri_run_eddy_batch_main "$DMRI_FAKE_RUNNER" first.yaml second.yaml
```

The fake runner appends NUL-safe or tab-delimited arguments to a capture file. Assert both invocations are exactly:

```text
--only-stage 05_eddy first.yaml
--only-stage 05_eddy second.yaml
```

Add no-argument and option-looking-argument tests; both must fail before invoking the runner. Paths containing spaces must remain one argument.

- [ ] **Step 2: Add failing continuation and aggregate-status tests**

Make the fake runner return `40` for the second config and `0` otherwise. Assert all three configs run sequentially, stdout contains:

```text
EDDY_BATCH_START config=first.yaml
EDDY_BATCH_RESULT config=second.yaml exit_code=40
EDDY_BATCH_FAILED count=1
```

and the batch returns `1`. Add an all-success case that returns `0` and reports `EDDY_BATCH_COMPLETE count=3`.

- [ ] **Step 3: Add failing package-audit expectations**

Update the expected executable inventory to contain:

```python
{"run_pipeline.sh", "run_eddy_batch.sh", "setup_rocky.sh"}
```

Run:

```bash
python -m pytest tests/test_wrappers.py tests/test_package_audit.py -q
```

Expected: the wrapper is missing and the executable allowlist rejects it.

- [ ] **Step 4: Implement a strict, sourceable Bash wrapper**

Create `run_eddy_batch.sh` with `set -uo pipefail` but do not use top-level `set -e`, because participant failures must be collected. The core loop should be equivalent to:

```bash
_dmri_run_eddy_batch_main() {
  local runner="$1"
  shift
  if (( $# == 0 )); then
    printf 'ERROR: at least one CONFIG.yaml is required\n' >&2
    return 2
  fi

  local total=$# failures=0 config status
  local -a failed_configs=()
  for config in "$@"; do
    if [[ "$config" == -* ]]; then
      printf 'ERROR: configuration arguments must be explicit paths: %s\n' "$config" >&2
      return 2
    fi
  done

  for config in "$@"; do
    printf 'EDDY_BATCH_START config=%s\n' "$config"
    if "$runner" --only-stage 05_eddy "$config"; then
      status=0
    else
      status=$?
      failures=$((failures + 1))
      failed_configs+=("$config:$status")
    fi
    printf 'EDDY_BATCH_RESULT config=%s exit_code=%s\n' "$config" "$status"
  done

  if (( failures == 0 )); then
    printf 'EDDY_BATCH_COMPLETE count=%s\n' "$total"
    return 0
  fi
  printf 'EDDY_BATCH_FAILED count=%s configs=%s\n' \
    "$failures" "${failed_configs[*]}" >&2
  return 1
}
```

Resolve the public runner with `SCRIPT_DIR` and an absolute sibling path. Do not parse YAML, use `eval`, glob inside the script, background commands, or suppress the underlying pipeline output.

- [ ] **Step 5: Mark the wrapper executable and allow only its exact path**

Run:

```bash
chmod 755 run_eddy_batch.sh
```

Add only `run_eddy_batch.sh` to `_ALLOWED_EXECUTABLES`; do not broaden the allowlist pattern.

- [ ] **Step 6: Run wrapper/package tests and commit**

Run:

```bash
python -m pytest tests/test_wrappers.py tests/test_package_audit.py -q
```

Expected: all tests pass, including the existing Darwin rejection of `run_pipeline.sh` and the sourceable fake-runner batch tests. The batch wrapper inherits the public Rocky gate by invoking `run_pipeline.sh` for every config.

Commit:

```bash
git add run_eddy_batch.sh src/dmri_pipeline/package_audit.py tests/test_wrappers.py tests/test_package_audit.py
git commit -m "feat: add sequential EDDY batch wrapper"
```

---

### Task 7: Document the split workflow, migration boundary, and release

**Files:**
- Modify: `README.md`
- Modify: `docs/PIPELINE.md`
- Modify: `docs/OUTPUTS.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `docs/REMOTE_VSCODE.md`
- Modify: `VERSION`
- Modify: `pyproject.toml`
- Modify: `tests/test_package_audit.py`
- Modify: `tests/test_state.py`

**Interfaces:**
- Produces: public `2.1.0` documentation and metadata for the bounded EDDY workflow.
- Preserves: Rocky-only execution, research-only warning, fixed scientific order, and existing full-pipeline quick start.

- [ ] **Step 1: Add failing documentation and version tests**

Require the public docs to contain the exact supported commands:

```text
./run_pipeline.sh --stop-after 04_bet config/subject.yaml
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml
./run_eddy_batch.sh config/subject-001.yaml config/subject-002.yaml
./run_pipeline.sh config/subject.yaml
```

Require `docs/OUTPUTS.md` to name `05_eddy/eddy_timing.json`, and require both `VERSION` and `project.version` to equal `2.1.0`. Update any stage-record version assertion in `tests/test_state.py` to `2.1.0`.

- [ ] **Step 2: Run package/state tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_package_audit.py tests/test_state.py -q
```

Expected: docs lack the new workflow and metadata still says `2.0.0`.

- [ ] **Step 3: Update the README and ordered pipeline contract**

Document the three-part workflow and these semantics:

- `--stop-after` is an ordered prefix and reports `PARTIAL_COMPLETE`.
- `--only-stage` validates exact-current upstream state and reports `STAGE_COMPLETE`.
- an exact-current selected stage skips safely;
- `--force-stage 05_eddy --only-stage 05_eddy` is the explicit rerun command and archives downstream results;
- normal invocation resumes after EDDY and continues from the first non-current downstream stage;
- no bounded mode bypasses QC, provenance, validation, or atomic promotion.

Keep the ordinary full command first or equally prominent for users who do not need the split workflow.

- [ ] **Step 4: Document outputs and timing interpretation**

In `docs/OUTPUTS.md`, list `eddy_timing.json` with the exact six-key schema and state:

- `eddy_command_seconds` includes EDDY's requested CNR-map and residual generation;
- `eddy_quad_seconds` measures the subsequent EDDY QUAD command;
- `stage_action_seconds` includes both commands plus stage-local postprocessing;
- values are monotonic wall durations in seconds and are not scientific QC thresholds.

State that a supervisor's approximate two-hour EDDY duration is plausible for a different command, image size, volume count, CPU, storage system, FSL release, or requested outputs, but is not a guarantee for this dataset or server.

- [ ] **Step 5: Document safe recovery and batch behaviour**

In `docs/TROUBLESHOOTING.md`, add:

- exact-current single-stage skip and intentional forced rerun examples;
- a warning that partial non-resumable `05_eddy.work.*` directories remain blocked and must follow the existing inspected recovery procedure;
- batch continuation semantics and final nonzero exit;
- no concurrent EDDY jobs by default;
- no deployment of upgraded code into a currently running subject.

Explicitly explain the source-evidence migration boundary: because `orchestrator.py` is hashed into stage records, records written by `2.0.0` are intentionally stale under `2.1.0`. Finish an active run before deployment. For intentional reprocessing, use a fresh `output_root` or an explicit force from the desired safe boundary; never manually edit `.stage_complete.json`.

- [ ] **Step 6: Add the Rocky `tmux` Remote SSH examples**

In `docs/REMOTE_VSCODE.md`, show all commands from a VS Code Remote terminal connected to Rocky:

```bash
tmux new -s dmri-eddy
cd /server/path/to/dmri_reproducible_pipeline
export DMRI_SOFTWARE_CONFIG=/server/private/path/dmri-rocky9.sh
./run_pipeline.sh --stop-after 04_bet config/subject.yaml
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml
```

For a cohort, show the explicit wrapper inside `tmux`, then detach with `Ctrl-b d` and reattach with `tmux attach -t dmri-eddy`. Reiterate that the Mac, SSHFS, and a local terminal must not execute the MRI jobs.

- [ ] **Step 7: Bump the public minor version**

Set:

```text
VERSION: 2.1.0
pyproject.toml project.version: 2.1.0
```

Update exact version expectations only; do not loosen them to prefix or range matches.

- [ ] **Step 8: Run documentation/package/state tests and commit**

Run:

```bash
python -m pytest tests/test_package_audit.py tests/test_state.py -q
```

Expected: all public inventory, documentation, workspace-safety, and version tests pass.

Commit:

```bash
git add README.md docs/PIPELINE.md docs/OUTPUTS.md docs/TROUBLESHOOTING.md docs/REMOTE_VSCODE.md VERSION pyproject.toml tests/test_package_audit.py tests/test_state.py
git commit -m "docs: publish split EDDY server workflow"
```

---

### Task 8: Complete regression and Rocky-server acceptance

**Files:**
- Modify only if a regression exposes a defect in the files already listed above.

**Interfaces:**
- Produces: one reviewable `2.1.0` change set with automated local evidence and explicit Rocky acceptance evidence.

- [ ] **Step 1: Run formatting/static checks already configured by the repository**

Inspect `pyproject.toml` for the configured tools, then run only those configured commands. At minimum run:

```bash
python -m compileall -q src tests
```

Expected: exit `0`; do not add a new formatter or linter as part of this feature.

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
python -m pytest -q
```

Expected: the complete suite passes. Fix root causes rather than weakening exact-current, timing, wrapper, or documentation assertions.

- [ ] **Step 3: Audit the final diff and package contents**

Run:

```bash
git status --short
git diff --check
git log --oneline --decorate -8
```

Confirm there are no MRI inputs/outputs, subject identifiers, server paths, secrets, software configuration files, timing fixtures from real participants, `.work.*` directories, or changed generated assets in the commit set.

- [ ] **Step 4: Deploy only after active server work has stopped**

On the Rocky server, first confirm no pipeline is using the repository or subject output. Pull or copy the committed release only after the current subject finishes. Do not hot-swap `orchestrator.py` during EDDY or any other stage.

- [ ] **Step 5: Run the Rocky preflight**

From the server repository in a Remote SSH terminal:

```bash
export DMRI_SOFTWARE_CONFIG=/server/private/path/dmri-rocky9.sh
./setup_rocky.sh --check
```

Expected: Rocky 9.7 x86_64, Conda, pinned Python environment, FSL 6.0.7.18 CPU EDDY, MATLAB R2025a/Optimization Toolbox/MEX, resources, and package checks all pass. This complete preflight is intentionally stricter than EDDY-only runtime discovery.

- [ ] **Step 6: Run a non-production bounded smoke case**

Using a dedicated test subject/output root, run inside `tmux`:

```bash
./run_pipeline.sh --stop-after 04_bet config/smoke-subject.yaml
./run_pipeline.sh --only-stage 05_eddy config/smoke-subject.yaml
./run_pipeline.sh --only-stage 05_eddy config/smoke-subject.yaml
./run_pipeline.sh config/smoke-subject.yaml
```

Expected:

1. the first command ends `PARTIAL_COMPLETE` at `04_bet`;
2. the first EDDY command ends `STAGE_COMPLETE` and creates validated `05_eddy/eddy_timing.json`;
3. the repeated EDDY-only command reports `05_eddy` as `SKIPPED` without rewriting it;
4. the normal command skips current upstream/EDDY stages and continues downstream;
5. no MATLAB discovery occurs in the EDDY-only invocation logs, while the complete resume discovers MATLAB before NODDI as required.

- [ ] **Step 7: Exercise batch continuation with test configurations**

Supply at least two safe test YAMLs, including one intentionally invalid upstream state:

```bash
./run_eddy_batch.sh config/smoke-current.yaml config/smoke-missing-upstream.yaml
```

Expected: both configs are attempted sequentially, the valid subject succeeds or skips, the invalid subject prints its underlying nonzero code, and the wrapper returns `1` with one-item failure summary.

- [ ] **Step 8: Inspect timing and final report evidence**

Verify `eddy_timing.json` contains finite nonnegative values and the report shows all three durations. Compare the complete EDDY duration with the supervisor's approximate two hours as observational evidence only; do not change acceptance or rerun solely because the duration differs.

- [ ] **Step 9: Record acceptance without committing participant data**

Record only generic pass/fail notes, package version, and test command results in the project issue or lab log. Do not commit subject IDs, absolute server paths, raw logs, MRI files, real timing JSON, or report outputs.
