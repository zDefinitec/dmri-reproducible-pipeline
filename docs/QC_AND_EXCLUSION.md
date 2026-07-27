# QC and exclusion

This is research QC, not clinical interpretation. Single-subject QC does not
estimate cohort-level fSI and must not be used for diagnosis.

The raw PA stripe metric classifies each volume:

- normal: cSI below 1.15;
- ambiguous: cSI from 1.15 through 1.25;
- high: cSI above 1.25.

Five or more high volumes produce `EXCLUDE` (exit 20). One or more ambiguous
volumes with `ambiguous_qc_reviewed: false` produce `HOLD_FOR_REVIEW` (exit
21). Fewer than five high volumes produce `INCLUDE_WITH_FLAGS`; reviewed
ambiguous volumes produce `INCLUDE_AFTER_REVIEW`. Include outcomes continue
with exit 0.

For a hold, inspect all three raw QC figures, anatomy, flagged one-based
volume numbers, shell grouping, and source images. Record the independent
human decision outside this package. To continue after genuine review, set
`analysis.ambiguous_qc_reviewed: true` and rerun normally. Do not force past
the gate. Exclusion is a data-quality outcome, not a software failure.

Later QC includes denoise/Gibbs/TOPUP/BET/EDDY comparisons, motion and
outlier plots, DTI/DKI/NODDI panels, atlas overlay, and a 4×4 overview.
Figures use common slices and robust finite display limits where relevant.
Review them with stage metrics and report warnings; a clean automatic status
does not establish scientific suitability.
