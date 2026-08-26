# Install on Rocky Linux

## Prerequisites

Use only a Rocky Linux 9.7 x86_64 server. Install Conda or Miniforge, a
compatible FSL distribution, MATLAB, MATLAB Optimization Toolbox, and a C
compiler configured with `mex -setup C`. Licences and installation of those
external products remain the recipient's responsibility.

FSL must provide `topup`, `applytopup`, `bet`, `fslmaths`, `eddy_quad`,
`flirt`, `fnirt`, `invwarp`, `applywarp`, and either `eddy_openmp` or `eddy`.
It must also provide the TOPUP/FNIRT configuration and standard FA files
checked by the script.

## Configure, create, and check the environment

From the package root:

```bash
cp config/software.rocky.example.sh /absolute/path/to/dmri-rocky9.sh
# Edit CONDA_EXE, FSLDIR, MATLAB_EXECUTABLE, DMRI_EXPECTED_FSL_VERSION,
# and DMRI_EXPECTED_MATLAB_VERSION in that private server file.
export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh
./setup_rocky.sh
./setup_rocky.sh --check
```

Copy the example to a private absolute path on the server; do not commit that
file. Edit all five required variables before exporting `DMRI_SOFTWARE_CONFIG`
in the current server shell. The first command updates the `dmri-repro`
environment from `environment.yml`, installs this package without dependency
resolution, and runs checks. `--check` is non-installing. Checks cover the
Rocky platform, exact Python versions, FSL tools/resources, MATLAB version,
Optimization Toolbox installation and licence, a real temporary MEX
compile/load/run, fixed atlas hashes, and available disk.

The MEX output must be `mexa64`. `tools.fsldir` and
`tools.matlab_executable` in a subject YAML remain higher-precedence,
subject-specific overrides. The runtime revalidates the final selected FSL and
MATLAB against the exact versions from the private software configuration and
runs the same MATLAB capability probe, including a real temporary MEX
compile/load/run; an invalid override is rejected without fallback.

Direct Python library callers that omit `DMRI_EXPECTED_FSL_VERSION` or
`DMRI_EXPECTED_MATLAB_VERSION` receive capability-only discovery. This is an
explicit testing/library behavior, not a public-wrapper mode: both public
wrappers require and export both expected-version values.

The setup script verifies but does not install or license FSL, MATLAB,
Optimization Toolbox, or compiler dependencies. The package compiles the
three required NIfTI MEX modules into a subject stage at runtime and never
adds binaries to this source tree. It uses CPU EDDY only. Resolve dependency
errors before rerunning.
