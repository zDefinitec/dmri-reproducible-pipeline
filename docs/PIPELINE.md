# Pipeline stages

The public stage order is fixed. Each stage validates its complete inputs,
writes into a private work directory, and validates outputs before promotion.
Filesystems with a native no-replace rename publish the final directory in one
operation. On NFS without that extension, the runner atomically reserves the
exact final name with an empty private directory, then atomically replaces its
own reservation with the validated work tree. Directory existence alone does
not mean completion; only an exact-current `.stage_complete.json` does.

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

## Optional split EDDY execution

The full command above is the ordinary workflow. A Rocky server operator can
instead divide an otherwise unchanged scientific order into an upstream prefix,
the EDDY stage, and normal downstream continuation. Every command below must
run inside a tmux session in a VS Code Remote terminal connected to Rocky:

```bash
./run_pipeline.sh --stop-after 04_bet config/subject.yaml
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml
./run_pipeline.sh config/subject.yaml
```

`--stop-after 04_bet` runs the ordered prefix through `04_bet` and reports
`PARTIAL_COMPLETE`. It does not change validation, QC, provenance, or atomic
promotion. `--only-stage 05_eddy` requires every upstream stage to be
exact-current and reports `STAGE_COMPLETE`; it likewise does not bypass QC,
provenance, validation, or atomic promotion. If the selected EDDY stage is
already exact-current, this single-stage command safely skips it and still
reports `STAGE_COMPLETE`.

To intentionally rerun EDDY, explicitly archive its result and every
downstream result before running it alone:

```bash
./run_pipeline.sh --force-stage 05_eddy --only-stage 05_eddy config/subject.yaml
```

After EDDY is exact-current, the normal command resumes with the first
non-current downstream stage. Do not use bounded execution to evade a QC,
provenance, validation, or promotion failure.

Run only one pipeline invocation per subject at a time, and do not run
`--force-stage` or invalidate that subject concurrently. This is part of the
NFS reservation safety model. An interruption during NFS promotion can leave
an empty noncurrent final directory plus the intact work directory; explicit
forcing archives both before rerunning that stage.

`--dry-run` validates and prints stage status and exact external argv without
writing stages. `--force-stage NAME` archives `NAME` and every downstream
stage. It is permitted with a selected run when the forced stage is inside the
selected execution range, including `--force-stage 05_eddy --only-stage
05_eddy`. It does not bypass input or output validation and must never be used
to conceal a QC or provenance failure.
