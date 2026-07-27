# Troubleshooting

## FSL discovery

Set `FSLDIR` or `tools.fsldir`, or put `topup` on `PATH`. Run
`./setup_macos.sh --check`. Missing TOPUP, BET, registration tools,
`eddy_openmp`/`eddy`, `eddy_quad`, configs, or the standard FA image is a
dependency error (exit 30). This macOS package uses CPU EDDY; it does not
configure CUDA EDDY. EDDY or EDDY QUAD process failure is exit 40, while
malformed scientific output is exit 50.

Odd slice counts select the alternate TOPUP configuration checked by setup.
Do not change `slice_axis`, PA/AP vectors, or `total_readout_time` merely to
silence TOPUP.

## MATLAB, Optimization Toolbox, and MEX

Set `MATLAB_EXECUTABLE` or `tools.matlab_executable` if MATLAB discovery
fails. The package requires a licensed Optimization Toolbox and a selected C
compiler. Run `mex -setup C`, then rerun `./setup_macos.sh --check`. Setup
performs a temporary compile/load/run; it does not install MATLAB, a compiler,
or a licence.

The NIfTI `file2mat`, `mat2file`, and `init` MEX modules are compiled locally
inside the NODDI stage. Never copy a MEX binary from another architecture.

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
