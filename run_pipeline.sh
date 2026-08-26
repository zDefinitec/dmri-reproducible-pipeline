#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCRIPT_DIR}/src"

fail() {
    echo "ERROR: $*" >&2
    exit 30
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/rocky_environment.sh"
check_rocky_platform
load_software_config

[[ "${CONDA_EXE}" == /* && -x "${CONDA_EXE}" ]] \
    || fail "configured CONDA_EXE must be an absolute executable path"
CONDA_BIN=${CONDA_EXE}

if ! "${CONDA_BIN}" run -n dmri-repro python -c 'import sys' \
    >/dev/null 2>&1
then
    echo "ERROR: required Conda environment 'dmri-repro' is unavailable" >&2
    exit 30
fi

exec "${CONDA_BIN}" run -n dmri-repro python -m dmri_pipeline.cli "$@"
