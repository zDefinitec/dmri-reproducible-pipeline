# VS Code Remote SSH workflow

Use VS Code **Remote - SSH** from the Mac only as the display and editing
client. The repository, data, interpreter, and computation remain on the Rocky
server. Saving an edited file in the Remote SSH window modifies the server file
directly; it does not make a local Mac copy.

## Connect and verify the server context

Keep connection details in the operator's local `~/.ssh/config`; do not commit
them to this repository. This placeholder-only example illustrates the alias
used below:

```sshconfig
Host dmri-rocky
    HostName SERVER_HOSTNAME
    User SERVER_USERNAME
    IdentityFile ~/.ssh/id_ed25519
```

In VS Code, choose **Remote - SSH: Connect to Host...**, select
`dmri-rocky`, then use **Open Folder** to open the repository on the Rocky
server. The status bar must show `SSH: dmri-rocky` before editing or running
commands.

### Mac-local connection check

Before opening the VS Code Remote window, check the alias from a Mac-local
terminal, then exit that shell:

```bash
ssh dmri-rocky
exit
```

### VS Code Remote integrated-terminal check

Once the status bar shows `SSH: dmri-rocky`, verify the remote context directly
in the VS Code integrated terminal. Do not start another SSH session there:

```bash
cat /etc/os-release
uname -m
pwd
```

The server must be Rocky Linux 9.7 on x86_64; confirm both the operating-system
release and `uname -m` output before proceeding. Input paths in
`config/subject.yaml` resolve on the Rocky server. Do not copy data to the Mac,
and do not use SSHFS for bulk MRI computation.

## Select the server Python interpreter

In the VS Code remote terminal, configure the operator-specific server software
file and run the standard preflight:

```bash
export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh
source "${DMRI_SOFTWARE_CONFIG}"
./setup_rocky.sh --check
./run_pipeline.sh --validate-only config/subject.yaml
./run_pipeline.sh --dry-run config/subject.yaml
```

`setup_rocky.sh` and `run_pipeline.sh` load this configuration only in their
own subprocesses, so source the private configuration in the interactive
remote terminal before discovering the installed server interpreter:

```bash
"${CONDA_EXE}" run -n dmri-repro which python
```

Run **Python: Select Interpreter** in the Remote SSH window and select the
returned server path. This selection is remote-workspace state; do not commit a
machine-specific absolute interpreter path in workspace settings.

## Protect long runs with tmux

There is no scheduler in this workflow. Start long pipeline work in `tmux` and
run the pipeline directly, without a pipe that could hide the pipeline exit
code. Run every command below in a VS Code Remote terminal connected to Rocky:

```bash
tmux new -s dmri-eddy
cd /server/path/to/dmri_reproducible_pipeline
export DMRI_SOFTWARE_CONFIG=/server/private/path/dmri-rocky9.sh
./run_pipeline.sh --stop-after 04_bet config/subject.yaml
./run_pipeline.sh --only-stage 05_eddy config/subject.yaml
# Later, resume remaining stages normally if required.
./run_pipeline.sh config/subject.yaml
```

For a cohort, run the sequential wrapper inside the same Rocky `tmux` session:

```bash
./run_eddy_batch.sh config/subject-001.yaml config/subject-002.yaml
```

Detach with `Ctrl-b d` and later reattach with:

```bash
tmux attach -t dmri-eddy
```

A VS Code disconnection does not end a process running inside `tmux`. Reconnect
to `SSH: dmri-rocky` and attach to the existing session to monitor it. The Mac,
SSHFS, and a local terminal must not execute the MRI jobs.
