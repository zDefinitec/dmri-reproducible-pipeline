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
when a chain stops.

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
