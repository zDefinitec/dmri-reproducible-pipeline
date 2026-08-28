# Split EDDY Execution Design

Date: 2026-08-28

## Objective

Expose EDDY as an independently runnable and cohort-batchable stage on the
supported Rocky Linux 9.7 x86_64 server, as requested by the scientific
supervisor. Preserve the existing scientific data flow, validation,
provenance, stage promotion, and normal resume behaviour.

The Mac remains only a VS Code Remote SSH client. All selected-stage commands,
FSL execution, input data, logs, stage records, and outputs remain on the Rocky
server.

## Operator workflow

The supported three-part workflow will be:

```bash
# 1. Prepare one subject through the corrected-b0 brain mask.
./run_pipeline.sh --stop-after 04_bet config/subject.yaml

# 2. Run only the subject's EDDY stage.
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml

# Or run EDDY sequentially for a cohort whose pre-EDDY stages are current.
./run_eddy_batch.sh config/subjects/*.yaml

# 3. Resume normally; current stages through EDDY are skipped.
./run_pipeline.sh config/subject.yaml
```

The normal unbounded command remains unchanged. A user who does not request a
stage boundary receives the existing complete pipeline behaviour.

## Command-line interface

Add two bounded-execution options to the existing CLI:

- `--stop-after STAGE` runs the normal ordered prefix ending at `STAGE`.
- `--only-stage STAGE` runs exactly one stage after proving that every upstream
  stage is exact-current.

Both values use the existing fixed `STAGE_ORDER` choices and are mutually
exclusive. They cannot be combined with `--validate-only` or `--dry-run`.
`--force-stage` is accepted with a bounded execution only when the forced
stage is inside the selected range. It retains the existing rule of archiving
that stage and every downstream stage, but execution stops at the requested
boundary. In particular, an intentional EDDY rerun is:

```bash
./run_pipeline.sh --force-stage 05_eddy --only-stage 05_eddy config/subject.yaml
```

`--only-stage` never manufactures or trusts a directory merely because it
exists. It validates every required upstream `.stage_complete.json`, the
current inputs, parameters, source evidence, software evidence, output hashes,
and the pre-denoise QC gate before running the selected stage. Missing, stale,
held, or excluded upstream state stops before external commands launch.

If the selected stage is already exact-current, it is reported as skipped.
Without explicit forcing, bounded execution never overwrites a completed
stage. When forcing is requested, downstream stages are archived before the
selected stage runs so stale downstream results cannot remain apparently
current.

Successful `--stop-after` execution returns status `PARTIAL_COMPLETE` and
successful `--only-stage` execution returns status `STAGE_COMPLETE`; both use
exit code zero and print the boundary stage. Existing exclusion, review-hold,
dependency, external-command, and validation exit codes remain unchanged.

## Dependency discovery

Bounded execution discovers only software required by the selected stages:

- the input audit and pre-denoise QC require no FSL or MATLAB discovery;
- preparation through `04_bet` requires FSL but not MATLAB;
- `05_eddy` requires FSL but not MATLAB;
- MATLAB discovery occurs only when an executed stage actually invokes or
  records MATLAB-dependent work, including NODDI.

Normal full-pipeline execution retains the existing strict FSL and MATLAB
requirements. This prevents an EDDY-only server job from failing because a
MATLAB licence is temporarily unavailable even though EDDY does not use
MATLAB.

## EDDY scientific command

The EDDY stage continues to consume:

- the full four-dimensional PA DWI after denoising and Gibbs correction;
- the cleaned mask derived from the mean TOPUP-corrected b0 images;
- the EDDY index and acquisition-parameter files;
- original canonical b-values and b-vectors;
- the TOPUP output basename.

It must not use the corrected-b0-only image as `--imain`, and it must not feed
an already EDDY-corrected DWI back into EDDY.

Keep the accepted current production options, including outlier replacement,
CNR maps, residuals, shelled-data declaration, CPU thread limit, and EDDY QUAD.
Add explicit `--fwhm=0` and `--flm=quadratic` arguments so the recorded command
can be compared directly with the supervisor's example. These are already the
current FSL defaults and therefore do not intentionally change the numerical
model.

Executable discovery remains version-aware. The Rocky server may expose
`eddy_openmp`, `eddy_cpu`, or the `eddy` launcher; the validated discovered CPU
executable is recorded instead of hard-coding a name copied from another FSL
installation.

## Runtime evidence

Record separate monotonic elapsed durations for:

1. the complete EDDY command, including requested CNR-map and residual
   generation;
2. the subsequent EDDY QUAD command;
3. their total wall time within the stage.

Store these finite, nonnegative values in a validated `eddy_timing.json` stage
output. Timing is operational evidence, not a scientific acceptance threshold:
the stage must not fail merely because it takes more or less than two hours.
The report states that `eddy_command_seconds` includes CNR/residual generation;
it does not claim to measure the supervisor's smaller example command. EDDY
QUAD time remains separately identifiable.

## Cohort batch wrapper

Add `run_eddy_batch.sh` as a thin server-side wrapper around the validated
single-subject entry point. It accepts one or more explicit YAML paths as
arguments and invokes:

```bash
./run_pipeline.sh --only-stage 05_eddy CONFIG.yaml
```

for each path sequentially. It does not discover subjects, parse YAML, rebuild
FSL commands, use `eval`, or run concurrent EDDY jobs. Sequential execution is
the safe default for predictable CPU, memory, and storage load on the server.

The wrapper prints a concise subject/config start and result line. A failed
subject is recorded and the remaining supplied configurations continue. The
wrapper exits zero only when every subject succeeds or is already current; it
exits nonzero after printing a final list of failed configurations otherwise.
The underlying per-subject exit code is printed beside each failure.

## State and failure behaviour

- No bounded mode bypasses input audit, QC decisions, output validation,
  software provenance, or atomic stage promotion.
- Existing final directories are never overwritten.
- A partial non-resumable EDDY work directory remains blocked and receives the
  same actionable recovery guidance as the normal pipeline.
- An interrupted batch can be rerun with the same arguments. Exact-current
  EDDY stages skip; incomplete or failed subjects are reported individually.
- Normal invocation after a successful EDDY-only run verifies and skips all
  current upstream stages before continuing downstream modelling.
- The batch wrapper does not hide a failed participant merely because later
  participants succeeded.

## Documentation

Update the README, ordered-pipeline documentation, troubleshooting guide, and
Remote SSH guide with the three-part server workflow. The documentation must
state that a supervisor's approximate two-hour runtime normally refers to the
core EDDY command and is not a guarantee across image dimensions, volume
counts, CPU models, storage, FSL releases, or optional QC products.

All examples run inside a Rocky-server `tmux` session from the VS Code Remote
terminal. No example runs MRI computation on the Mac or across SSHFS.

## Verification

Automated tests will cover:

- CLI acceptance and mutual-exclusion errors for the bounded options;
- prefix execution through `04_bet` without MATLAB discovery;
- exact `05_eddy` execution with exact-current upstream stages;
- refusal of EDDY-only execution for missing, stale, held, or excluded
  upstream state;
- exact-current EDDY skip behaviour;
- unchanged normal full-pipeline behaviour and exit codes;
- explicit `--fwhm=0` and `--flm=quadratic` command arguments;
- validated timing JSON with separate EDDY-command, EDDY-QUAD, and total
  durations;
- batch success, mixed failure, argument safety, sequential ordering, and
  rerun behaviour;
- wrapper/package audit updates for the new intended executable script;
- Rocky wrapper tests proving commands still execute through the configured
  server Conda/FSL environment.

Server acceptance requires a representative subject to complete the three
commands in a `tmux` session, followed by inspection of the corrected DWI,
rotated b-vectors, movement/outlier evidence, EDDY QUAD report, timing record,
and normal downstream resume.

## Non-goals

- changing denoising, Gibbs, TOPUP, BET, EDDY, DTI, DKI, NODDI, atlas, or
  summary algorithms;
- promising a two-hour EDDY runtime;
- removing current EDDY QC products merely to match a runtime estimate;
- introducing GPU/CUDA EDDY, Slurm, PBS, containers, or concurrent cohort
  scheduling;
- running or monitoring the Rocky server from the local Mac checkout;
- accepting raw unvalidated external EDDY outputs as current pipeline state.
