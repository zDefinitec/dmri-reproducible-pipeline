# Reproducible Rocky Linux diffusion MRI pipeline

This package runs a single-subject PA/AP diffusion MRI workflow with
deterministic validation, resumable stages, QC, DTI, two DKI implementations,
NODDI, a historical 48-label JHU atlas summary, and a final report.

**Research only. Not for clinical diagnosis, treatment, or clinical
interpretation.** The bundled JHU atlas is under the main FSL
non-commercial licence. Review [third-party notices](licenses/THIRD_PARTY_NOTICES.md)
before redistribution or use.

## Supported system

- Rocky Linux 9.7 x86_64 server
- a private absolute-path server software configuration exported through
  `DMRI_SOFTWARE_CONFIG`
- Conda or Miniforge
- FSL with TOPUP, BET, CPU EDDY, EDDY QUAD, and registration tools
- MATLAB with Optimization Toolbox and a working C MEX compiler that emits
  `mexa64`
- enough runtime memory and disk space for uncompressed NODDI working data

The setup script installs only the pinned Python environment and this Python
package. It verifies, but does not install or license, FSL, MATLAB,
Optimization Toolbox, or compiler dependencies. VS Code Remote SSH is the
supported operator workflow, but it is not required to execute the pipeline.

## Quick start

```bash
cp config/subject.example.yaml config/subject.yaml
export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh
./setup_rocky.sh
./run_pipeline.sh config/subject.yaml
```

Edit `config/subject.yaml` first. Supply PA DWI, PA b-values and b-vectors, one
or more AP b0 volumes, the PA and AP phase-encoding vectors, and the
scanner-derived `total_readout_time`. Acquisition values are user-supplied;
the package does not infer them from DICOM or sidecars.

## Optional split EDDY workflow (Rocky server only)

The ordinary full command above remains the standard workflow. When EDDY must
be scheduled separately, every command below must run inside a tmux session
in a VS Code Remote terminal connected to Rocky:

```bash
# Prepare the ordered upstream prefix.
./run_pipeline.sh --stop-after 04_bet config/subject.yaml

# Run the validated EDDY stage after its upstream stages are exact-current.
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml

# Resume the normal pipeline, which continues at the first non-current stage
# after EDDY.
./run_pipeline.sh config/subject.yaml
```

For a cohort, use the sequential wrapper in that same Rocky `tmux` session; it
runs no concurrent EDDY jobs by default:

```bash
./run_eddy_batch.sh config/subject-001.yaml config/subject-002.yaml
```

See the [ordered workflow contract](docs/PIPELINE.md),
[recovery guidance](docs/TROUBLESHOOTING.md), and
[Remote SSH instructions](docs/REMOTE_VSCODE.md). Do not run MRI computation
on the Mac, through SSHFS, or from a Mac-local terminal.

## Optional preflight

Optional preflight commands are:

```bash
./setup_rocky.sh --check
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
Confirm actual free disk with `./setup_rocky.sh --check` and monitor the first
representative subject before scheduling a cohort.

## Documentation

- [Rocky Linux installation](docs/INSTALL_ROCKY.md)
- [input and configuration contract](docs/INPUTS.md)
- [ordered pipeline](docs/PIPELINE.md)
- [outputs](docs/OUTPUTS.md)
- [QC and exclusion](docs/QC_AND_EXCLUSION.md)
- [troubleshooting](docs/TROUBLESHOOTING.md)
- [VS Code Remote SSH operator workflow](docs/REMOTE_VSCODE.md)
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
