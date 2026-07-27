#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCRIPT_DIR}/src"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_BIN=${CONDA_EXE}
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=$(command -v conda)
else
    echo "ERROR: Conda/Miniforge is required; conda was not found" >&2
    exit 30
fi

if ! "${CONDA_BIN}" run -n dmri-repro python -c 'import sys' \
    >/dev/null 2>&1
then
    echo "ERROR: required Conda environment 'dmri-repro' is unavailable" >&2
    exit 30
fi

exec "${CONDA_BIN}" run -n dmri-repro python -m dmri_pipeline.cli "$@"
