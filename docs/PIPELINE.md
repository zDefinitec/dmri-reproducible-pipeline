# Pipeline stages

The public stage order is fixed. Each stage validates its complete inputs,
writes into a private work directory, validates outputs, and atomically
promotes a final directory with `.stage_complete.json`.

1. `00_input_audit` — input identity, NIfTI, gradients, acquisition, and
   software-independent validation.
2. `00_pre_denoise_motion_qc` — raw anatomy/cSI figures, metrics, and binding
   include/hold/exclude gate.
3. `01_denoise` — PA/AP MP-PCA denoising, sigma maps, mask bootstrap, and
   metrics.
4. `02_gibbs` — PA/AP Gibbs-ringing removal and metrics.
5. `03_topup` — PA/AP b0 extraction, acqparams/index, TOPUP, corrected b0, and
   field/motion validation.
6. `04_bet` — brain extraction and deterministic largest-component mask
   cleanup.
7. `05_eddy` — CPU EDDY correction, rotated b-vectors, motion/outlier/CNR
   products, and EDDY QUAD numeric evidence.
8. `06_dti` — DTI scalar/vector maps using the configured maximum b-value.
9. `07_dki` — DIPY DKI maps.
10. `07_dki_direct` — independent Henrique direct-DKI maps.
11. `08_noddi` — local MEX preparation, resumable MATLAB workers, merge, maps,
    and error-code metrics.
12. `09_jhu_48roi` — nearest-neighbour warp of the fixed historical JHU
    48-label atlas.
13. `10_summary` — six masks, 17 global metrics, and 48 ROI rows.
14. `qc` — validated per-stage and overview PNG figures plus manifest.
15. `report` — sanitized deterministic JSON and Markdown final report.

Normal invocation resumes stages whose inputs, parameters, software evidence,
outputs, and manifest are exact-current:

```bash
./run_pipeline.sh config/subject.yaml
```

`--dry-run` validates and prints stage status and exact external argv without
writing stages. `--force-stage NAME` is valid only for a normal run. It
archives `NAME` and every downstream stage; it does not bypass input or
output validation and must never be used to conceal a QC or provenance
failure.
