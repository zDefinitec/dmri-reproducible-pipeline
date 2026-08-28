# Troubleshooting

## FSL discovery

Confirm that the current Rocky Linux 9.7 x86_64 server shell exports
`DMRI_SOFTWARE_CONFIG` to a private absolute configuration file. Set `FSLDIR`
there, unless a subject-specific `tools.fsldir` override is required. Run
`./setup_rocky.sh --check`. Missing TOPUP, BET, registration tools,
`eddy_openmp`/`eddy`, an `eddy_cpu` backend required by the `eddy` launcher,
`eddy_quad`, configs, or the standard FA image is a dependency error (exit
30). This package selects CPU EDDY directly; it does not configure or audit
unused CUDA EDDY backends. EDDY or EDDY QUAD process failure is exit 40, while
malformed scientific output is exit 50. Resolve dependency errors before
rerunning.

Odd slice counts select the alternate TOPUP configuration checked by setup.
Do not change `slice_axis`, PA/AP vectors, or `total_readout_time` merely to
silence TOPUP.

## MATLAB, Optimization Toolbox, and MEX

Set `MATLAB_EXECUTABLE` in the private software configuration, or use the
higher-precedence subject-specific `tools.matlab_executable` override, if
MATLAB discovery fails. The package requires a licensed Optimization Toolbox,
a selected C compiler, and MEX output named `mexa64`. Run `mex -setup C`, then
rerun `./setup_rocky.sh --check`. Setup performs a temporary compile/load/run;
it verifies but does not install MATLAB, a compiler, or a licence.

The NIfTI `file2mat`, `mat2file`, and `init` MEX modules are compiled locally
inside the NODDI stage. Never copy a MEX binary from another architecture.

## NFS stage promotion

Some NFS4 mounts do not implement Linux `renameat2(RENAME_NOREPLACE)`. The
runner falls back to an exact-name empty-directory reservation and refuses a
final path already present when that reservation is acquired. Run only one
invocation per subject, and do not concurrently force or invalidate it; NFS
cannot conditionally rename against a reservation removed by another process
with permission to mutate the subject directory. If an interruption leaves
both an empty noncurrent final directory and the intact `.work` directory,
resolve the interruption and use `--force-stage NAME` to archive both before
restarting that stage.

## Split EDDY recovery and batch behaviour

After a successful upstream prefix, an exact-current `05_eddy` can be checked
or safely skipped without rerunning it:

```bash
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml
```

For an intentional EDDY rerun, use the explicit bounded force command. It
archives EDDY and downstream results before executing the selected stage:

```bash
./run_pipeline.sh --force-stage 05_eddy --only-stage 05_eddy config/subject.yaml
```

A partial, non-resumable `05_eddy.work.*` directory remains blocked. Inspect
the failure and preserve the evidence, then follow the existing inspected
recovery procedure: resolve the cause and use the explicit force command from
the required safe boundary. Do not delete or edit stage evidence by hand.

For cohorts, run the wrapper on Rocky after each subject's upstream stages are
exact-current:

```bash
./run_eddy_batch.sh config/subject-001.yaml config/subject-002.yaml
```

The wrapper processes configuration files sequentially (no concurrent EDDY
jobs by default), continues after an individual subject fails, and exits
nonzero after the final subject if any subject failed. Review every reported
failure before rerunning a selected subject.

## Version 2.1.0 source-evidence migration boundary

`orchestrator.py` is hashed into stage records. Therefore, stage records
written by 2.0.0 are intentionally stale under 2.1.0. Finish an active subject
run before deploying upgraded code; never deploy an upgrade into a currently
running subject. For intentional reprocessing under 2.1.0, use a fresh
`output_root` or explicitly force from the desired safe boundary. Never
manually edit `.stage_complete.json`.

## NODDI workers

Normal reruns validate preparation and completed worker hashes, resume valid
workers, and merge only the exact expected worker partition. Preserve the
stage directory after interruption. Do not edit checkpoints or final worker
MAT files. Use `--force-stage 08_noddi` only when intentionally discarding
the NODDI stage and all downstream atlas, summary, QC, and report results.

## QC hold or exclusion

`HOLD_FOR_REVIEW` requires human review of ambiguous volumes. Record review
with `ambiguous_qc_reviewed: true` and rerun normally. `EXCLUDE` means at
least five high volumes and is not overridden by the review flag. Do not use
`--force-stage` to bypass either result.

## Historical JHU 48 versus current 50 labels

The bundled resource is the historical FSL `fsl-5_0_4` JHU ICBM-DTI-81
48-label variant. It intentionally differs from current 50-label variants.
Labels 45/46 are right/left uncinate; labels 31/32 are sagittal stratum names
that include IFOF. Do not substitute a current atlas or use non-nearest
interpolation.

## Exit codes and safe forcing

- 0: complete, validated, dry-run, or included;
- 2: CLI/config/input error;
- 20: excluded by the binding stripe gate;
- 21: hold for review;
- 30: missing dependency or invalid packaged resource;
- 40: external command failure;
- 50: stage state or scientific output validation failure.

Resolve the underlying cause first. A normal rerun resumes exact-current
stages. `--force-stage NAME` archives that stage and every later stage; it
cannot be combined with `--validate-only` or `--dry-run`.
