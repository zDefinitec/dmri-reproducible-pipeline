# Install on macOS

## Prerequisites

Use macOS on Apple Silicon (`arm64`) or Intel (`x86_64`). Install Conda or
Miniforge, a compatible FSL distribution, MATLAB, MATLAB Optimization
Toolbox, and a C compiler configured with `mex -setup C`. Licences and
installation of those external products remain the recipient's
responsibility.

FSL must provide `topup`, `applytopup`, `bet`, `fslmaths`, `eddy_quad`,
`flirt`, `fnirt`, `invwarp`, `applywarp`, and either `eddy_openmp` or
`eddy`. It must also provide the TOPUP/FNIRT configuration and standard FA
files checked by the script.

## Create and check the environment

From the package root:

```bash
./setup_macos.sh
./setup_macos.sh --check
```

The first command updates the `dmri-repro` environment from
`environment.yml`, installs this package without dependency resolution, and
runs checks. `--check` is non-installing. Checks cover architecture, exact
Python versions, FSL tools/resources, MATLAB version, Optimization Toolbox
installation and licence, a real temporary MEX compile/load/run, fixed atlas
hashes, and available disk.

Set `FSLDIR` or make `topup` discoverable. Set `MATLAB_EXECUTABLE` when
`matlab` is not on `PATH` and no suitable application exists under
`/Applications`. These discovery overrides may instead be placed in the YAML
under `tools`.

The package does not download FSL, MATLAB, Optimization Toolbox, or
proprietary licences. It compiles the three required NIfTI MEX modules into a
subject stage at runtime and never adds binaries to this source tree.
