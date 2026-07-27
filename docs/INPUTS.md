# Inputs and configuration

Start from `config/subject.example.yaml`. Relative paths are resolved from the
configuration file's directory.

```yaml
subject_id: SUBJECT_ID
inputs:
  dwi_pa: inputs/pa_dwi.nii.gz
  bvals: inputs/pa_dwi.bval
  bvecs: inputs/pa_dwi.bvec
  b0_ap: inputs/ap_b0.nii.gz
output_root: ../outputs
acquisition:
  pa_vector: [0, -1, 0]
  ap_vector: [0, 1, 0]
  total_readout_time: REPLACE_WITH_SCANNER_VALUE_SECONDS
  slice_axis: 2
analysis:
  dti_max_b: 1200.0
  noddi_workers: auto
  ambiguous_qc_reviewed: false
tools:
  fsldir: null
  matlab_executable: null
```

`subject_id` is a portable identifier, not a path. The PA input is a finite
4D NIfTI. AP may be finite 3D or 4D. B-value count must equal PA volumes.
B-vectors must be finite 3×N (N×3 is canonicalized); non-b0 vectors and
accepted b0 conventions are norm-validated without rewriting the input.
At least one PA b0 (`b < 50`) is required.

`pa_vector` and `ap_vector` are three integral FSL phase-encoding vectors and
must be opposite. `total_readout_time` is a finite positive scanner-derived
value in seconds; its placeholder is intentionally invalid until replaced.
`slice_axis` is 0, 1, or 2. The pipeline never guesses PA, AP, or readout
time.

`dti_max_b` selects the DTI subset. `noddi_workers` is `auto` or a positive
integer bounded by CPU and memory. Keep `ambiguous_qc_reviewed: false` until a
human has reviewed every ambiguous raw volume; set it to `true` to record
that review and resume.

All inputs must be distinct readable regular files with safe path components.
Symlinks, hard-link aliases, special files, changing files, nonfinite data,
shape/grid/count mismatches, and unsafe output roots fail closed.

Validate without creating stages:

```bash
./run_pipeline.sh --validate-only config/subject.yaml
```
