# Rocky Linux and VS Code Remote Development Design

Date: 2026-08-26

## Objective

Convert the current macOS-only diffusion MRI package into a Rocky Linux
9.7, x86_64 package. The source code, imaging data, dependencies, working
files, and results will reside on one Rocky Linux server. A macOS computer
will provide only the Visual Studio Code user interface and will connect to
the server with VS Code Remote SSH.

The scientific pipeline and its validation rules remain unchanged unless a
platform dependency requires a narrowly scoped compatibility change.

## Supported deployment

- Client: a supported macOS release with Visual Studio Code, an OpenSSH
  client, and the Remote - SSH extension.
- Server: Rocky Linux 9.7 on x86_64, running directly rather than through a
  cluster scheduler.
- Execution: all Python, FSL, MATLAB, MEX compilation, tests, and pipeline
  commands run on the Rocky server.
- Storage: source, input data, stage state, logs, and outputs remain on the
  server. Bulk computation must not access MRI data through SSHFS.
- Connectivity: SSH authentication and network access are configured by the
  server administrator or operator. Credentials and private keys are never
  stored in this repository.

Other Linux distributions, macOS execution, ARM processors, schedulers, and
containers are outside the supported scope of this version.

## Architecture

The deployment follows the useful part of CBIG's environment model: large or
licensed applications are installed independently in versioned server
locations, while one project-specific configuration exposes their paths to a
single setup and runtime interface.

The layers are:

1. The macOS client runs the VS Code user interface and SSH client.
2. VS Code Remote SSH installs and manages its server component in the
   operator's Rocky account and opens the repository directory in place.
3. A local-to-the-server software configuration declares the exact Conda,
   FSL, and MATLAB locations.
4. A Rocky setup script creates or updates the pinned Python environment and
   performs fail-closed dependency probes.
5. The existing pipeline wrapper launches the Python orchestrator on the
   server with the same software configuration.

No source synchronization layer is introduced. Saving a file in the Remote
SSH window writes that server file directly.

## Server software integration

The project requires only the external tools it actually uses:

- Conda or Miniforge;
- FSL with TOPUP, BET, CPU EDDY, EDDY QUAD, and the required registration
  tools and resources;
- MATLAB with Optimization Toolbox;
- a MATLAB-supported C compiler selected for MEX compilation.

The project will include a version-controlled example configuration without
machine-specific values. The operator will copy it to a private server
location and set paths such as `FSLDIR`, `MATLAB_EXECUTABLE`, and optionally
`CONDA_EXE`. A single environment variable will identify that private file
for both setup and execution. The repository will not source arbitrary
shell startup files or modify `.bashrc` automatically.

The setup process will:

1. read `/etc/os-release` and require Rocky Linux 9.7;
2. require `uname -m` to report `x86_64`;
3. locate Conda, FSL, and MATLAB from explicit configuration;
4. update the pinned `dmri-repro` Conda environment and install the local
   Python package without dependency resolution;
5. verify exact Python package versions;
6. verify every required FSL executable, configuration file, and standard
   image;
7. run the existing MATLAB capability probe, including Optimization Toolbox
   installation, licence availability, selected C compiler, and a real
   compile/load/run MEX test;
8. validate packaged resource hashes and available disk space.

Missing paths, wrong versions, failed licences, or incomplete tool
capabilities remain fatal dependency errors. Unlike CBIG's general setup,
the package will not continue after only printing a version warning.

## Platform-specific code changes

- Replace `setup_macos.sh` with `setup_rocky.sh` and remove macOS discovery
  branches.
- Replace the macOS `sysctl hw.memsize` probe with a bounded parser for
  `MemTotal` in `/proc/meminfo`.
- Discover MATLAB through explicit configuration or the server `PATH`.
  `/Applications/MATLAB_R*.app` discovery is removed.
- Retain the existing Linux `renameat2` no-replace implementation. The Darwin
  filesystem primitive may remain solely so pre-migration unit tests can run
  from a Mac checkout; the public setup and runtime entry points must still
  reject macOS, and Darwin is not a supported execution platform.
- Rename macOS-specific installation documentation and package metadata.
- Update the package audit executable allowlist, required-document list, and
  wrapper tests for the Rocky filenames.
- Keep `environment.yml` as the authoritative pinned Python environment.
- Keep CPU EDDY selection (`eddy_openmp`, then `eddy`) unless a later
  requirement explicitly introduces CUDA EDDY.

## VS Code remote workspace

The repository will contain non-secret workspace guidance:

- `.vscode/extensions.json` will recommend the Python and Remote SSH-related
  extensions needed for this project.
- Workspace settings may select the server-side `dmri-repro` interpreter by
  command or documented selection steps, but must not contain a username,
  hostname, password, private key path, or fixed home directory.
- Installation documentation will show how to test `ssh user@host`, connect
  with `Remote-SSH: Connect to Host`, open the remote repository folder, and
  confirm that the integrated terminal reports Rocky Linux.
- Server extensions are installed in the Remote SSH context rather than used
  to run local macOS Python against remote data.

Long-running work will be launched inside `tmux` from the VS Code remote
terminal. The documented workflow will create a named session, run the
pipeline there, detach safely, and reattach after a client or network
disconnect. Existing stage-resume validation remains the recovery mechanism
after an actual server-side process interruption.

## Data and command flow

1. The operator connects from the Mac to the Rocky host with Remote SSH.
2. VS Code opens the repository directory on the server.
3. The operator edits a subject YAML whose paths resolve on the server.
4. The operator runs `setup_rocky.sh --check` in the remote terminal.
5. The operator runs input validation before starting computation.
6. The operator starts or attaches to a `tmux` session and launches the
   pipeline wrapper.
7. The wrapper runs the Conda Python process on Rocky; Python invokes FSL and
   MATLAB through validated absolute paths.
8. Outputs, stage records, QC artifacts, and reports are written to server
   storage and inspected through the remote VS Code window.

No stage transfers source or imaging data to the Mac.

## Error handling and security

- A missing software configuration, unsupported OS, wrong architecture, or
  incomplete dependency fails before scientific stages begin.
- The software configuration is operator-owned and excluded from public
  package exports. The example contains placeholders only.
- SSH configuration remains in the operator's macOS `~/.ssh` directory.
- Input directories should be read-only where server permissions allow;
  output directories must be writable by the operator.
- Commands will not embed passwords or licence secrets in arguments, logs,
  YAML, or workspace settings.
- The wrapper will preserve pipeline exit codes so remote use does not hide
  exclusion, review-hold, dependency, external-command, or validation
  failures.

## Verification

Automated tests will cover:

- Rocky 9.7 and x86_64 platform acceptance and rejection cases;
- `/proc/meminfo` parsing, including missing, malformed, zero, and bounded
  positive values;
- explicit Linux MATLAB discovery and error messages;
- relocation-safe setup and runtime wrappers using fake Conda, FSL, and
  MATLAB executables;
- updated package-audit allowlists and required documentation;
- absence of remaining supported-path references to macOS or Darwin;
- the existing full Python test suite.

Server acceptance will additionally require:

1. successful VS Code Remote SSH connection;
2. a remote terminal showing Rocky Linux 9.7 and x86_64;
3. successful `setup_rocky.sh --check` using the real server software;
4. successful configuration validation and dry-run on a server-side example;
5. a representative subject run, with versions and capability evidence
   recorded in the generated audit and stage records.

## Non-goals

- Installing or licensing MATLAB, Optimization Toolbox, or FSL automatically;
- copying MRI data to the Mac;
- adding unrelated CBIG applications such as FreeSurfer, SPM, AFNI, ANTs, or
  Connectome Workbench;
- supporting Docker, Apptainer, Singularity, Slurm, PBS, GPU EDDY, or multiple
  Linux distributions in this release;
- promising bit-identical results across old macOS and new Rocky executions.
