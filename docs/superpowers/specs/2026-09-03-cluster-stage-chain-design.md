# Cluster Stage Chain Design

Date: 2026-09-03

## Objective

Add three independently scheduled execution groups to the existing
single-subject diffusion MRI pipeline:

1. TOPUP preparation and correction;
2. BET and EDDY;
3. tensor models, NODDI, atlas summarization, QC, and reporting.

The groups will run as separate CPU jobs through `CBIG_pbsubmit`. A successful
TOPUP job submits the EDDY job, and a successful EDDY job submits the NODDI
and post-processing job. A failed, excluded, or review-held job stops the
chain.

This change separates scheduling and resource allocation. It does not copy or
replace scientific implementations, relax validation, or change scientific
selection criteria.

## Chosen approach

The pipeline will retain one Python orchestrator and one implementation of
every scientific stage. A bounded stage-group option will select which slice
of the existing plan is allowed to execute. Thin shell wrappers will connect
those bounded runs to `CBIG_pbsubmit`.

This is preferred to either:

- copying TOPUP, EDDY, or NODDI code into standalone implementations, which
  would allow the full and scheduled workflows to diverge; or
- invoking the unbounded pipeline in every job and relying only on checkpoint
  skipping, which would let the first job continue into stages for which it
  did not request resources.

## Stage-group boundaries

The public group names and inclusive pipeline ranges are fixed:

| Group | First stage | Last stage | Contents |
| --- | --- | --- | --- |
| `topup` | `00_input_audit` | `03_topup` | input audit, pre-denoise motion QC, denoise, Gibbs correction, TOPUP |
| `eddy` | `04_bet` | `05_eddy` | BET mask generation and EDDY |
| `noddi` | `06_dti` | `report` | DTI, both DKI paths, NODDI, JHU registration, summary, QC, report |

The names refer to scheduler resource groups rather than claiming that each
job contains only one command.

## Pipeline command interface

The existing entry point gains one optional argument:

```text
./run_pipeline.sh --stage-group {topup,eddy,noddi} CONFIG.yaml
```

Without `--stage-group`, the existing full-pipeline behavior remains
unchanged.

A normal grouped run:

- validates the subject configuration and constructs the same authoritative
  stage plan as a full run;
- verifies that every stage before the selected group is exact-current;
- executes or skips exact-current stages only inside the selected group;
- does not execute, invalidate, or write downstream group stages; and
- returns zero only when the complete selected group is valid and complete.

If an upstream stage is absent, stale, unsafe, or incomplete, the grouped run
fails with the first affected stage and the group that must be run first. It
does not silently repair an upstream group.

`--force-stage` remains an explicit destructive-to-derived-state recovery
operation. When combined with `--stage-group`, the named stage must belong to
that group. The existing dependency rule is retained: the named stage and all
later stages are archived, after which only the selected group's range is
executed. This prevents stale downstream results from surviving a forced
upstream recomputation. Without `--stage-group`, `--force-stage` retains its
current full-pipeline behavior.

`--dry-run --stage-group GROUP` reports prerequisite status and the bounded
stage plan without writing results. `--validate-only` remains a whole-subject
preflight and does not accept `--stage-group`, because it performs no stages.

The grouped completion result will identify the subject, group, and output
root. Existing full-run result statuses and exit codes remain compatible.

## Cluster-facing files

The repository will add:

```text
scripts/cluster/common.sh
scripts/cluster/submit_subject_chain.sh
scripts/cluster/run_topup_subject.sh
scripts/cluster/run_eddy_subject.sh
scripts/cluster/run_noddi_subject.sh
config/cluster.example.sh
docs/CLUSTER.md
```

The wrappers are scheduling adapters only. They call `run_pipeline.sh` with a
stage group and never invoke FSL or MATLAB scientific commands directly.

The operator copies `config/cluster.example.sh` to
`config/cluster.local.sh`. The local file is ignored by Git. It contains the
machine-specific absolute paths and initial resource requests:

```bash
CBIG_PBSUBMIT=/absolute/path/to/CBIG_pbsubmit
DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh
CLUSTER_RUN_ROOT=/absolute/path/under/the/operators/home/dmri_cluster_runs

TOPUP_WALLTIME=04:00:00
TOPUP_MEM=16G
TOPUP_NCPUS=4

EDDY_WALLTIME=04:00:00
EDDY_MEM=16G
EDDY_NCPUS=8

NODDI_WALLTIME=24:00:00
NODDI_MEM=32G
NODDI_NCPUS=8
```

These values are conservative starting points, not scientific constants.
Cluster-chain execution requires `analysis.noddi_workers` to be an explicit
positive integer so worker selection cannot accidentally reflect head-node
rather than allocated compute-node resources. `NODDI_NCPUS` must be at least
that value. Before submission, a small Python helper reuses the package's
existing `load_config` validation to emit the subject ID, subject output path,
and configured NODDI worker count. The shell code does not parse YAML itself.
The launcher rejects an invalid or missing resource value, an automatic NODDI
worker setting, or a worker count larger than the requested CPUs before
submission.

The committed example contains placeholders only. Passwords, keys, licences,
and private server paths must not enter version control.

## Submission and data flow

The operator starts a chain with absolute configuration paths:

```text
scripts/cluster/submit_subject_chain.sh \
    /absolute/path/to/subject.yaml \
    /absolute/path/to/cluster.local.sh
```

The launcher creates a filesystem-safe unique `chain_id` and a chain
directory below `CLUSTER_RUN_ROOT`. It then submits the TOPUP wrapper with the
TOPUP resource request.

Each job receives the same three immutable inputs:

- the absolute subject YAML path;
- the absolute private cluster configuration path;
- the `chain_id`.

The flow is:

```text
launcher
  -> submit TOPUP job
       -> run --stage-group topup
       -> on exit 0 only, submit EDDY job
            -> run --stage-group eddy
            -> on exit 0 only, submit NODDI job
                 -> run --stage-group noddi
                 -> mark the chain complete
```

The shared helper validates paths and resource values, builds the single
string required by `CBIG_pbsubmit` with shell-safe quoting, and supplies
explicit `-jobout` and `-joberr` paths. It must not use `eval`.

Recovery can begin at a selected group:

```text
scripts/cluster/submit_subject_chain.sh --start-at eddy SUBJECT.yaml CLUSTER.local.sh
```

Starting at EDDY or NODDI still performs the Python prerequisite validation.
It cannot bypass missing or stale upstream evidence.

## Chain state and duplicate prevention

Every chain directory stores small scheduling records separately from the
scientific output directories. Records include the chain ID, subject config
path, group, timestamps, attempted submission command arguments, wrapper exit
status, and whether the next group was submitted.

Before a wrapper submits its successor, it takes an exclusive lock in its
chain directory and checks the successor-submitted marker. The marker is
written only after `CBIG_pbsubmit` returns success. Re-entering the same
wrapper with the same chain ID therefore does not intentionally submit the
successor twice.

A new launcher invocation creates a new chain ID and represents an explicit
new scheduling attempt. The design does not claim to detect unrelated jobs
submitted outside these wrappers.

`CBIG_pbsubmit` does not expose a dependency flag in its public interface. For
CPU jobs submitted away from the head node, its current implementation
forwards submission to `headnode`; the wrappers rely on that behavior for
success-triggered nested submission. A zero exit from `CBIG_pbsubmit` means
the submission wrapper returned successfully, not that the scheduled job
later completed. Its stdout and stderr are therefore retained as chain
evidence.

## Exit and failure behavior

The wrapper preserves the pipeline exit status. In particular:

- `0`: selected group complete; submission of the next group is permitted;
- `20`: subject excluded by the pre-denoise QC gate; stop;
- `21`: ambiguous QC requires review; stop;
- dependency, external-command, output-validation, stage-state, and other
  nonzero failures: stop.

If submission of a successor fails, the current scientific group remains
complete but the chain is marked `submission_failed` and exits nonzero. The
operator can rerun the same chain wrapper to retry submission without
recomputing exact-current scientific stages.

Scheduler termination, walltime exhaustion, or host failure cannot be
mistaken for scientific success because the next group is submitted only
after the bounded pipeline process returns zero.

Existing checkpoint and NODDI worker-resume behavior remain authoritative.
No result directory, `.invalidated` directory, or user file is deleted by the
cluster wrappers.

## Compatibility and security

- The existing ungrouped command remains the default and continues to run the
  full pipeline.
- Stage implementations, inputs, output paths, validators, hashes, and
  scientific thresholds remain shared with the full pipeline.
- Group execution must use the existing subject lock, preventing concurrent
  mutation of one subject output root.
- All machine-specific paths must be absolute. The scripts reject newline and
  other unsafe control characters in values used to build a submitted
  command.
- Runtime and log directories must resolve below the operator-configured
  `CLUSTER_RUN_ROOT`; wrappers do not use system `/tmp`.
- No `sudo` operation is introduced.

## Verification

Python tests will cover:

- the exact, exhaustive, non-overlapping group boundaries;
- unchanged stage order and unchanged ungrouped execution;
- bounded execution and exact-current skipping inside each group;
- rejection of missing, stale, partial, or unsafe upstream state;
- rejection of a forced stage outside the selected group;
- downstream invalidation after an explicitly forced stage;
- bounded dry-run output and no-write behavior; and
- preservation of exclusion, review-hold, and error exit statuses.

Shell-wrapper tests will use fake pipeline and `CBIG_pbsubmit` executables to
cover:

- exact resource flags for each group;
- safe propagation of absolute config paths and the chain ID;
- TOPUP-to-EDDY and EDDY-to-NODDI submission only after exit zero;
- stopping on every nonzero status;
- logging of submission output;
- duplicate-successor prevention within one chain;
- recovery beginning at each group; and
- rejection of malformed, relative, missing, or private-placeholder
  configuration values.

The relevant unit tests and then the full Python suite will run before the
implementation is declared complete. No real cluster job or full MRI pipeline
run is required for the local implementation tests. A later server acceptance
check will use one subject configuration, inspect the three scheduler logs,
and confirm that existing stage records remain valid across job boundaries.

## Non-goals

- Duplicating or forking scientific stage code;
- changing TOPUP, BET, EDDY, DTI, DKI, NODDI, atlas, summary, or QC methods;
- adding GPU EDDY;
- supporting schedulers other than the supplied `CBIG_pbsubmit` interface;
- automatically choosing resources from scanner dimensions;
- automatically resubmitting a failed scientific stage; or
- modifying or deleting existing result and `.invalidated` directories.
