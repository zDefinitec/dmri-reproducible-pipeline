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

_dmri_run_pipeline_main() {
    local release_file=$1 uname_bin=$2 conda_bin
    shift 2
    check_rocky_platform "${release_file}" "${uname_bin}"
    load_software_config

    [[ "${CONDA_EXE}" == /* && -x "${CONDA_EXE}" ]] \
        || fail "configured CONDA_EXE must be an absolute executable path"
    conda_bin=${CONDA_EXE}

    if ! "${conda_bin}" run -n dmri-repro python -c 'import sys' \
        >/dev/null 2>&1
    then
        echo "ERROR: required Conda environment 'dmri-repro' is unavailable" >&2
        return 30
    fi

    exec "${conda_bin}" run --no-capture-output -n dmri-repro \
        python -u -m dmri_pipeline.cli "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _dmri_run_pipeline_main /etc/os-release /usr/bin/uname "$@"
fi
