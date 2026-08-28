# Outputs

Outputs live under `<output_root>/<subject_id>/`. The 15 final stage
directories use the names in `PIPELINE.md`; `.work` is transaction space and
is never part of a complete result. Every complete stage has a
`.stage_complete.json` record binding inputs, parameters, software,
resources, and output hashes. A directory without a valid exact-current record
is incomplete, including an empty NFS promotion reservation.

Important products include:

- raw stripe QC: `stripe_metrics.csv`, `stripe_decision.json`,
  `automatic_summary.txt`, and three review PNGs;
- preprocessing: denoised/Gibbs PA and AP images, TOPUP inputs/corrections,
  cleaned `nodif_brain_mask.nii.gz`, EDDY-corrected DWI, rotated b-vectors,
  motion/outlier/CNR evidence, and sanitized EDDY QUAD metrics;
- `06_dti`: `FA.nii.gz`, `MD.nii.gz`, `AD.nii.gz`, `RD.nii.gz`,
  `V1.nii.gz`, and `dti_metrics.json`;
- `07_dki`: `FA.nii.gz`, `MD.nii.gz`, `AD.nii.gz`, `RD.nii.gz`,
  `V1.nii.gz`, `MK.nii.gz`, `AK.nii.gz`, `RK.nii.gz`, and
  `dki_metrics.json`;
- `07_dki_direct`: `MD.nii.gz`, `MK.nii.gz`, `S0.nii.gz`, and
  `dki_direct_metrics.json`;
- NODDI: `NODDI_odi.nii`, `NODDI_ficvf.nii`, `NODDI_fiso.nii`, the other
  validated maps `NODDI_kappa.nii`, `NODDI_fmin.nii`,
  `NODDI_error_code.nii`, `NODDI_fibredirs_xvec.nii`,
  `NODDI_fibredirs_yvec.nii`, and `NODDI_fibredirs_zvec.nii`, plus
  `NODDI_params.mat` and `noddi_metrics.json`; worker checkpoints and MEX
  binaries remain stage-internal evidence;
- atlas: the nearest-neighbour 48-label subject-space atlas;
- summary: `<subject>_global_metrics.csv`,
  `<subject>_JHU_48ROI_metrics.csv`, and `<subject>_summary.json`;
- QC: 17 named PNGs (16 non-overview figures and one overview) plus
  `qc_manifest.json`;
- report: `<subject>_run_summary.json`, `<subject>_analysis_report.md`, and
  `<subject>_QC_report.pdf`.

## EDDY timing evidence

`05_eddy/eddy_timing.json` is required provenance for the complete EDDY stage.
It has exactly these six keys:

```json
{
  "schema_version": 1,
  "eddy_command_seconds": 0.0,
  "eddy_quad_seconds": 0.0,
  "stage_action_seconds": 0.0,
  "eddy_command_includes_cnr_maps": true,
  "eddy_command_includes_residuals": true
}
```

`eddy_command_seconds` includes EDDY's requested CNR-map and residual
generation. `eddy_quad_seconds` measures the subsequent EDDY QUAD command.
`stage_action_seconds` includes both commands and stage-local postprocessing.
All three values are monotonic wall durations in seconds; they are
observational evidence, not scientific QC thresholds.

A supervisor's approximate two-hour EDDY duration can be plausible for a
different command, image size, volume count, CPU, storage system, FSL release,
or requested outputs. It is not a guarantee for this dataset or this server.

Model maps are finite after documented replacement rules. Atlas data must be
integral with nonzero labels exactly 1–48. Summary mask counts, ROI voxel
totals, QC figure dimensions/hashes, and relative report links are validated
before stage promotion.

The report intentionally omits absolute paths, URIs, credentials, work paths,
and nonfinite JSON values. Retain the complete stage directories, not just
selected maps, when reproducibility evidence is required.
