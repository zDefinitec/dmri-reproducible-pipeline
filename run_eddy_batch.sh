#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

_dmri_run_eddy_batch_main() {
    local runner=$1
    shift
    if (( $# == 0 )); then
        printf 'ERROR: at least one CONFIG.yaml is required\n' >&2
        return 2
    fi

    local total=$# failures=0 config status
    local -a failed_configs=()
    for config in "$@"; do
        if [[ "$config" == -* ]]; then
            printf 'ERROR: configuration arguments must be explicit paths: %s\n' "$config" >&2
            return 2
        fi
    done

    for config in "$@"; do
        printf 'EDDY_BATCH_START config=%s\n' "$config"
        if "$runner" --only-stage 05_eddy "$config"; then
            status=0
        else
            status=$?
            failures=$((failures + 1))
            failed_configs+=("$config:$status")
        fi
        printf 'EDDY_BATCH_RESULT config=%s exit_code=%s\n' "$config" "$status"
    done

    if (( failures == 0 )); then
        printf 'EDDY_BATCH_COMPLETE count=%s\n' "$total"
        return 0
    fi
    printf 'EDDY_BATCH_FAILED count=%s configs=%s\n' \
        "$failures" "${failed_configs[*]}" >&2
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _dmri_run_eddy_batch_main "${SCRIPT_DIR}/run_pipeline.sh" "$@"
fi
