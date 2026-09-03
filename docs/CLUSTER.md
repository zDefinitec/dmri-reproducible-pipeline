# Cluster submission and recovery

This procedure schedules the existing pipeline as three CPU jobs: TOPUP,
EDDY, then NODDI. The cluster scripts are scheduling adapters only; they do
not run scientific tools themselves.

## Before submitting

Submit from the cluster head node with the repository available at the same
absolute location to every scheduled job. Confirm that `CBIG_pbsubmit` can
submit a simple permitted CPU job from that node, the configured software has
already passed `./setup_rocky.sh --check`, and the subject configuration uses
absolute input and output paths reachable by compute nodes.

Create the private scheduler configuration with restrictive permissions. It
is intentionally ignored by Git:

```bash
umask 077
cp config/cluster.example.sh config/cluster.local.sh
chmod 600 config/cluster.local.sh
```

Edit `config/cluster.local.sh` on the head node. Set the machine-specific
absolute paths for `CBIG_PBSUBMIT`, `DMRI_SOFTWARE_CONFIG`, and
`CLUSTER_RUN_ROOT`, plus walltime, memory, and CPU requests for TOPUP, EDDY,
and NODDI. Keep credentials, licence details, and private paths out of the
repository. `CLUSTER_RUN_ROOT` is the durable location for scheduler records;
it must be writable from the head node and all compute nodes.

Cluster execution requires an explicit positive integer in the subject YAML:

```yaml
analysis:
  noddi_workers: 4
```

Do not rely on automatic worker selection: it could observe the head node
rather than the allocated worker. Set `NODDI_NCPUS` in the private cluster
configuration to at least `analysis.noddi_workers`.

## Submit a chain

Use absolute paths when invoking the launcher. From the repository root, a
full chain is:

```bash
scripts/cluster/submit_subject_chain.sh \
  "$(pwd -P)/config/subject.yaml" \
  "$(pwd -P)/config/cluster.local.sh"
```

The launcher creates a new chain directory and submits TOPUP. A successful
TOPUP job submits EDDY; a successful EDDY job submits NODDI. To resume a
deliberately selected boundary after verifying its prerequisites, use:

```bash
scripts/cluster/submit_subject_chain.sh --start-at eddy \
  "$(pwd -P)/config/subject.yaml" \
  "$(pwd -P)/config/cluster.local.sh"
```

Valid `--start-at` values are `topup`, `eddy`, and `noddi`. Starting later
does not bypass the pipeline's prerequisite and stage-evidence validation.

## Monitor and tune resources

For each chain, scheduler stdout and stderr are stored at:

```text
CLUSTER_RUN_ROOT/chain-*/logs/topup.out and topup.err
CLUSTER_RUN_ROOT/chain-*/logs/eddy.out and eddy.err
CLUSTER_RUN_ROOT/chain-*/logs/noddi.out and noddi.err
```

The same chain directory contains immutable launch inputs, timestamps,
per-group status and exit-status records, submission arguments, and scheduler
submission stdout/stderr under `submissions/`. Inspect the `status` file first
when a chain stops. Before each worker starts, it re-reads the subject
configuration and requires its subject ID, resolved subject output, and NODDI
worker count to match those immutable launch records exactly.

Tune future requests from completed PBS `.STATS` files: compare elapsed time,
peak memory, and allocated CPUs for each group, then adjust only the matching
`*_WALLTIME`, `*_MEM`, and `*_NCPUS` values in the private configuration.
Keep `NODDI_NCPUS` at or above the explicit NODDI worker count. Treat the
first representative completed chains as measurements, not proof that every
acquisition will have the same demand.

`CBIG_pbsubmit` returning zero means only that the submission command
accepted the request. It does not prove that the queued job later started,
ran successfully, or produced valid pipeline output. Confirm scheduler state,
the chain records, and the pipeline logs.

## Exit codes and recovery

The worker preserves the bounded pipeline's exit status:

- `0`: selected group completed; the next group may be submitted.
- `20`: the pre-denoise QC gate excluded the subject; later groups stop.
- `21`: the QC gate requires documented review; later groups stop.
- `2`: configuration or input validation failed.
- `30`: a required dependency or cluster-wrapper preflight failed.
- `40`: an external command failed.
- `50`: output validation, stage state, or another pipeline processing error
  occurred.

If a group failed, correct the reported cause and rerun the appropriate
launcher command. If the scientific group completed but successor submission
failed, rerun that same worker command with the recorded immutable arguments
to retry only scheduler submission. Do not delete outputs, chain directories,
or stage evidence as a recovery step. The pipeline's existing resumable stage
state determines what must be recomputed; use `--start-at` only after checking
the recorded chain state and prerequisites.

### Stale submission locks

Initial and successor submission are fail-closed. The launcher holds a
subject-scoped guard at
`.subject-submission-locks/SUBJECT_OUTPUT_HASH.lock` under `CLUSTER_RUN_ROOT`;
it releases the guard only after the initial scheduler result and every chain
record are durable. A process killed after scheduler acceptance, or a durable
record failure after acceptance, leaves that guard in place so a new chain for
the same subject output cannot submit a duplicate job. Successor workers use
`.eddy.submission.lock` or `.noddi.submission.lock` in the chain directory.

Each lock's `owner` record identifies the subject or chain, requested group,
scheduler job name when allocated, host, wrapper PID, and acquisition time. A
dead PID alone does not prove that the scheduler rejected the job. An initial
guard with `state=allocating_chain` can result from abrupt death before a chain
or scheduler call existed, but still requires process and scheduler
reconciliation before removal. Abrupt death in the tiny interval between the
atomic lock-directory creation and publication of `owner` can instead leave an
empty lock. Treat an empty lock as uncertain too: identify the host and process
from scheduler and system logs and confirm that no matching submission reached
the scheduler before removing it.

Never blindly delete or supersede one of these locks. First inspect its `owner`
record, the matching chain's `*.submitted` marker and `status`, and every
`submissions/GROUP.*` record. Then reconcile the recorded job name
`dmri_GROUP_CHAIN_ID` against scheduler history and the live queue. If that job
is queued or running, leave the lock in place and do not resubmit it. If the
job was accepted, wait for its terminal scheduler state and reconcile that
exact job against its chain logs, worker exit/status records, and scientific
stage evidence. Repair any invalid record target only after identifying why
the durable write failed. Once the accepted job has a known terminal outcome
and cannot be duplicated, preserve the reconciliation evidence with the chain;
then the exact stale `owner` file and its now-empty lock directory may be
removed. Base any later recovery chain on the reconciled scientific state, not
on a replacement submission of the already accepted job.

If scheduler history proves that no matching job was accepted, and the
recorded host/PID is no longer live, the same exact-owner removal is safe. The
empty `.subject-submission-locks` parent may then be removed. If scheduler
acceptance or terminal outcome remains uncertain, leave the subject or chain
locked and do not launch a replacement `--start-at` chain until reconciliation
is complete.
