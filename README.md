# Reproducible macOS diffusion MRI pipeline

This package runs a single-subject PA/AP diffusion MRI workflow with
deterministic validation, resumable stages, QC, DTI, two DKI implementations,
NODDI, a historical 48-label JHU atlas summary, and a final report.

**Research only. Not for clinical diagnosis, treatment, or clinical
interpretation.** The bundled JHU atlas is under the main FSL
non-commercial licence. Review [third-party notices](licenses/THIRD_PARTY_NOTICES.md)
before redistribution or use.

## Supported system

- macOS on Apple Silicon (`arm64`) or Intel (`x86_64`)
- Conda or Miniforge
- FSL with TOPUP, BET, EDDY, EDDY QUAD, and registration tools
- MATLAB with Optimization Toolbox and a working local C MEX compiler
- enough runtime memory and disk space for uncompressed NODDI working data

The setup script installs only the pinned Python environment and this Python
package. It does not install or license FSL, MATLAB, Optimization Toolbox, or a
C compiler.

## Quick start

```bash
cp config/subject.example.yaml config/subject.yaml
./setup_macos.sh
./run_pipeline.sh config/subject.yaml
```

Edit `config/subject.yaml` first. Supply PA DWI, PA b-values and b-vectors, one
or more AP b0 volumes, the PA and AP phase-encoding vectors, and the
scanner-derived `total_readout_time`. Acquisition values are user-supplied;
the package does not infer them from DICOM or sidecars.

## Optional preflight

Optional preflight commands are:

```bash
./setup_macos.sh --check
./run_pipeline.sh --validate-only config/subject.yaml
```

Normal invocation resumes exact-current stages. `--dry-run` prints stage and
external-command plans without writing stages. `--force-stage NAME` archives
that stage and every later stage before a normal rerun; use it only after
understanding the dependency cascade.

The pre-denoise stripe gate can return `HOLD_FOR_REVIEW`,
`INCLUDE_WITH_FLAGS`, or `EXCLUDE`. An ambiguous case proceeds only after the
operator records review by changing `analysis.ambiguous_qc_reviewed` to
`true`; exclusions do not run later scientific stages.

## Runtime, memory, and disk planning

For a roughly 100–200-volume, 2 mm whole-brain protocol, use 4–24 hours and
20–80 GiB of free disk as a conservative starting allowance. These are
planning ranges, not guarantees: matrix size, brain-mask voxel count, volume
count, FSL/MATLAB versions, compiler, CPU generation, thermal limits, and
storage speed can move a run well outside them. NODDI is usually the dominant
runtime and its uncompressed work files are usually the dominant disk use.

Plan at least 16 GiB memory; 32 GiB or more is preferable for parallel NODDI.
Automatic selection permits at most eight workers, leaves two CPUs free, and
budgets 8 GiB of installed memory per worker. Lower
`analysis.noddi_workers` when the machine is shared or memory constrained.
Confirm actual free disk with `./setup_macos.sh --check` and monitor the first
representative subject before scheduling a cohort.

## Documentation

- [macOS installation](docs/INSTALL_MACOS.md)
- [input and configuration contract](docs/INPUTS.md)
- [ordered pipeline](docs/PIPELINE.md)
- [outputs](docs/OUTPUTS.md)
- [QC and exclusion](docs/QC_AND_EXCLUSION.md)
- [troubleshooting](docs/TROUBLESHOOTING.md)
- [third-party attribution](licenses/THIRD_PARTY_NOTICES.md)

## Reproducibility limits

The package pins Python versions, validates inputs and outputs, hashes stage
evidence, records software identities, and ships only source plus one fixed
atlas. This supports auditable reruns; it does not promise bit-identical
numerical output across materially different operating systems, CPU
architectures, FSL releases, MATLAB releases, compilers, or proprietary
toolbox versions.

There is no project-wide open-source licence or implied permission grant in
this distribution. Each bundled third-party component retains its own terms,
notices, source evidence, and hashes. Package-authored files are not covered
by a newly invented copyright or licence statement.

## Scientific references

- MP-PCA denoising: Veraart et al.,
  [Denoising of diffusion MRI using random matrix theory](https://doi.org/10.1016/j.neuroimage.2016.08.016).
- Local subvoxel-shift Gibbs correction: Kellner et al.,
  [Gibbs-ringing artifact removal](https://doi.org/10.1002/mrm.26054).
- Reverse-phase susceptibility correction: Andersson, Skare, and Ashburner,
  [TOPUP method](https://doi.org/10.1016/S1053-8119(03)00336-7).
- Integrated eddy-current/motion correction: Andersson and Sotiropoulos,
  [EDDY method](https://doi.org/10.1016/j.neuroimage.2015.10.019).
- Diffusion kurtosis imaging: Jensen et al.,
  [DKI definition](https://doi.org/10.1002/mrm.20508).
- NODDI: Zhang et al.,
  [practical in-vivo NODDI](https://doi.org/10.1016/j.neuroimage.2012.03.072).
- Direct-DKI procedure/source:
  [RafaelNH/CamCAN-dMRI-study](https://github.com/RafaelNH/CamCAN-dMRI-study).
- JHU ICBM-DTI-81 atlas: Mori et al.,
  [stereotaxic white-matter atlas](https://doi.org/10.1016/j.neuroimage.2007.12.035),
  and the [official FSL atlas documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html).
