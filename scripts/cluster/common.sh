#!/usr/bin/env bash
set -euo pipefail

CLUSTER_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(CDPATH= cd -- "${CLUSTER_SCRIPT_DIR}/../.." && pwd -P)

dmri_fail() {
    echo "ERROR: $*" >&2
    exit 30
}

dmri_reject_control_characters() {
    local name=$1 value=$2
    if [[ "${value}" == *[[:cntrl:]]* ]]; then
        dmri_fail "${name} must not contain control characters"
    fi
}

dmri_require_value() {
    local name=$1 value=$2
    [[ -n "${value}" ]] || dmri_fail "${name} is required"
    dmri_reject_control_characters "${name}" "${value}"
}

dmri_reject_placeholder() {
    local name=$1 value=$2
    case "${value}" in
        *REPLACE_WITH*|/absolute/path/*)
            dmri_fail "${name} contains a placeholder value"
            ;;
    esac
}

dmri_require_absolute_file() {
    local name=$1 value=$2
    dmri_require_value "${name}" "${value}"
    dmri_reject_placeholder "${name}" "${value}"
    [[ "${value}" == /* ]] || dmri_fail "${name} must be an absolute path"
    [[ -f "${value}" ]] || dmri_fail "${name} must name a regular file"
}

dmri_require_absolute_executable() {
    local name=$1 value=$2
    dmri_require_absolute_file "${name}" "${value}"
    [[ -x "${value}" ]] || dmri_fail "${name} must name an executable file"
}

dmri_require_absolute_directory_path() {
    local name=$1 value=$2
    dmri_require_value "${name}" "${value}"
    dmri_reject_placeholder "${name}" "${value}"
    [[ "${value}" == /* ]] || dmri_fail "${name} must be an absolute path"
}

dmri_validate_walltime() {
    local name=$1 value=$2
    dmri_require_value "${name}" "${value}"
    [[ "${value}" =~ ^[0-9][0-9]:[0-5][0-9]:[0-5][0-9]$ ]] \
        || dmri_fail "${name} must use HH:MM:SS"
}

dmri_validate_memory() {
    local name=$1 value=$2
    dmri_require_value "${name}" "${value}"
    [[ "${value}" =~ ^[1-9][0-9]*[KMGTP]([iI]?[Bb]?)?$ ]] \
        || dmri_fail "${name} must be a memory value such as 16G"
}

dmri_validate_positive_integer() {
    local name=$1 value=$2
    dmri_require_value "${name}" "${value}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || dmri_fail "${name} must be a positive integer"
}

dmri_validate_input_paths() {
    local subject_config=$1 cluster_config=$2
    dmri_require_absolute_file "subject configuration" "${subject_config}"
    dmri_require_absolute_file "cluster configuration" "${cluster_config}"
}

dmri_load_cluster_config() {
    local cluster_config=$1 key
    local required_keys=(
        CBIG_PBSUBMIT DMRI_SOFTWARE_CONFIG CLUSTER_RUN_ROOT
        TOPUP_WALLTIME TOPUP_MEM TOPUP_NCPUS
        EDDY_WALLTIME EDDY_MEM EDDY_NCPUS
        NODDI_WALLTIME NODDI_MEM NODDI_NCPUS
    )
    for key in "${required_keys[@]}"; do
        unset "${key}"
    done
    # shellcheck disable=SC1090
    source "${cluster_config}"
    for key in "${required_keys[@]}"; do
        dmri_require_value "${key}" "${!key-}"
    done

    dmri_require_absolute_executable "CBIG_PBSUBMIT" "${CBIG_PBSUBMIT}"
    dmri_require_absolute_file "DMRI_SOFTWARE_CONFIG" "${DMRI_SOFTWARE_CONFIG}"
    dmri_require_absolute_directory_path "CLUSTER_RUN_ROOT" "${CLUSTER_RUN_ROOT}"
    dmri_validate_walltime "TOPUP_WALLTIME" "${TOPUP_WALLTIME}"
    dmri_validate_memory "TOPUP_MEM" "${TOPUP_MEM}"
    dmri_validate_positive_integer "TOPUP_NCPUS" "${TOPUP_NCPUS}"
    dmri_validate_walltime "EDDY_WALLTIME" "${EDDY_WALLTIME}"
    dmri_validate_memory "EDDY_MEM" "${EDDY_MEM}"
    dmri_validate_positive_integer "EDDY_NCPUS" "${EDDY_NCPUS}"
    dmri_validate_walltime "NODDI_WALLTIME" "${NODDI_WALLTIME}"
    dmri_validate_memory "NODDI_MEM" "${NODDI_MEM}"
    dmri_validate_positive_integer "NODDI_NCPUS" "${NODDI_NCPUS}"
}

dmri_configured_conda() {
    local configured_conda
    configured_conda=$(
        unset CONDA_EXE
        # shellcheck disable=SC1090
        source "${DMRI_SOFTWARE_CONFIG}"
        printf '%s' "${CONDA_EXE-}"
    )
    dmri_require_absolute_executable "configured CONDA_EXE" "${configured_conda}"
    printf '%s\n' "${configured_conda}"
}

dmri_read_subject_context() {
    local subject_config=$1 conda_exe context_json parsed_context
    conda_exe=$(dmri_configured_conda)
    if ! context_json=$(
        DMRI_SOFTWARE_CONFIG="${DMRI_SOFTWARE_CONFIG}" \
            "${REPO_ROOT}/run_pipeline.sh" --print-cluster-context "${subject_config}"
    ); then
        dmri_fail "could not obtain validated cluster context"
    fi
    if ! parsed_context=$(
        DMRI_CLUSTER_CONTEXT_JSON="${context_json}" \
            "${conda_exe}" run --no-capture-output -n dmri-repro python -c '
import json
import os
import sys

def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(30)

try:
    payload = json.loads(os.environ["DMRI_CLUSTER_CONTEXT_JSON"])
except (KeyError, json.JSONDecodeError):
    fail("cluster context must be one JSON object")
if not isinstance(payload, dict):
    fail("cluster context must be one JSON object")
subject_id = payload.get("subject_id")
subject_output = payload.get("subject_output")
workers = payload.get("noddi_workers")
for name, value in (("subject_id", subject_id), ("subject_output", subject_output)):
    if not isinstance(value, str) or not value:
        fail("%s must be a non-empty string" % name)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        fail("%s must not contain control characters" % name)
if not subject_output.startswith("/"):
    fail("subject_output must be an absolute path")
if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
    fail("noddi_workers must be a positive integer")
print("%s\t%s\t%d" % (subject_id, subject_output, workers))
'
    ); then
        dmri_fail "invalid cluster context"
    fi
    IFS=$'\t' read -r SUBJECT_ID SUBJECT_OUTPUT NODDI_WORKERS <<< "${parsed_context}"
    dmri_require_value "subject_id" "${SUBJECT_ID}"
    dmri_require_value "subject_output" "${SUBJECT_OUTPUT}"
    dmri_validate_positive_integer "noddi_workers" "${NODDI_WORKERS}"
}

dmri_resource_for_group() {
    local group=$1
    case "${group}" in
        topup)
            RESOURCE_WALLTIME=${TOPUP_WALLTIME}
            RESOURCE_MEM=${TOPUP_MEM}
            RESOURCE_NCPUS=${TOPUP_NCPUS}
            ;;
        eddy)
            RESOURCE_WALLTIME=${EDDY_WALLTIME}
            RESOURCE_MEM=${EDDY_MEM}
            RESOURCE_NCPUS=${EDDY_NCPUS}
            ;;
        noddi)
            RESOURCE_WALLTIME=${NODDI_WALLTIME}
            RESOURCE_MEM=${NODDI_MEM}
            RESOURCE_NCPUS=${NODDI_NCPUS}
            ;;
        *)
            dmri_fail "unknown stage group: ${group}"
            ;;
    esac
}

dmri_shell_join() {
    local argument quoted command=""
    for argument in "$@"; do
        printf -v quoted '%q' "${argument}"
        if [[ -n "${command}" ]]; then
            command="${command} "
        fi
        command="${command}${quoted}"
    done
    printf '%s\n' "${command}"
}

dmri_write_record() {
    local target=$1 value=$2 temporary="${1}.tmp.$$"
    printf '%s\n' "${value}" > "${temporary}"
    mv -f -- "${temporary}" "${target}"
}

dmri_require_safe_chain_target() {
    local target=$1
    [[ ! -L "${target}" ]] \
        || dmri_fail "chain path must resolve below CLUSTER_RUN_ROOT without symbolic links: ${target}"
}

dmri_submit_group() {
    local group=$1 subject_config=$2 cluster_config=$3 chain_id=$4 chain_dir=$5
    local wrapper command status argument
    local submit_arguments=()
    dmri_resource_for_group "${group}"
    wrapper="${CLUSTER_SCRIPT_DIR}/run_${group}_subject.sh"
    dmri_require_absolute_executable "${group} worker wrapper" "${wrapper}"
    command=$(dmri_shell_join "${wrapper}" "${subject_config}" "${cluster_config}" "${chain_id}")
    submit_arguments=(
        -cmd "${command}"
        -walltime "${RESOURCE_WALLTIME}"
        -mem "${RESOURCE_MEM}"
        -ncpus "${RESOURCE_NCPUS}"
        -name "dmri_${group}_${chain_id}"
        -jobout "${chain_dir}/logs/${group}.out"
        -joberr "${chain_dir}/logs/${group}.err"
    )
    dmri_require_safe_chain_target "${chain_dir}/logs/${group}.out"
    dmri_require_safe_chain_target "${chain_dir}/logs/${group}.err"
    dmri_require_safe_chain_target "${chain_dir}/submissions/${group}.argv"
    dmri_require_safe_chain_target "${chain_dir}/submissions/${group}.stdout"
    dmri_require_safe_chain_target "${chain_dir}/submissions/${group}.stderr"
    dmri_require_safe_chain_target "${chain_dir}/submissions/${group}.exit_status"
    : > "${chain_dir}/submissions/${group}.argv"
    for argument in "${submit_arguments[@]}"; do
        printf '%q\n' "${argument}" >> "${chain_dir}/submissions/${group}.argv"
    done
    if "${CBIG_PBSUBMIT}" "${submit_arguments[@]}" \
        > "${chain_dir}/submissions/${group}.stdout" \
        2> "${chain_dir}/submissions/${group}.stderr"
    then
        status=0
    else
        status=$?
    fi
    dmri_write_record "${chain_dir}/submissions/${group}.exit_status" "${status}"
    if (( status == 0 )); then
        dmri_write_record "${chain_dir}/${group}.submitted" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    fi
    return "${status}"
}

dmri_validate_chain_id() {
    local chain_id=$1
    dmri_require_value "chain ID" "${chain_id}"
    [[ "${chain_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || dmri_fail "chain ID is not filesystem-safe"
}

dmri_resolve_chain_directory() {
    local name=$1 expected=$2 resolved
    [[ -d "${expected}" ]] || dmri_fail "${name} does not exist: ${expected}"
    [[ ! -L "${expected}" ]] \
        || dmri_fail "${name} must resolve below CLUSTER_RUN_ROOT without symbolic links"
    resolved=$(CDPATH= cd -- "${expected}" && pwd -P)
    case "${resolved}" in
        "${CLUSTER_RUN_ROOT}"/*) ;;
        *) dmri_fail "${name} must resolve below CLUSTER_RUN_ROOT" ;;
    esac
    [[ "${resolved}" == "${expected}" ]] \
        || dmri_fail "${name} must resolve below CLUSTER_RUN_ROOT without symbolic links"
    printf '%s\n' "${resolved}"
}

dmri_load_chain() {
    local subject_config=$1 cluster_config=$2 chain_id=$3 stored_subject stored_cluster
    dmri_validate_input_paths "${subject_config}" "${cluster_config}"
    dmri_load_cluster_config "${cluster_config}"
    dmri_validate_chain_id "${chain_id}"
    mkdir -p -- "${CLUSTER_RUN_ROOT}"
    CLUSTER_RUN_ROOT=$(CDPATH= cd -- "${CLUSTER_RUN_ROOT}" && pwd -P)
    CHAIN_DIR="${CLUSTER_RUN_ROOT}/${chain_id}"
    CHAIN_DIR=$(dmri_resolve_chain_directory "chain directory" "${CHAIN_DIR}")
    dmri_resolve_chain_directory "chain logs directory" "${CHAIN_DIR}/logs" >/dev/null
    dmri_resolve_chain_directory \
        "chain submissions directory" "${CHAIN_DIR}/submissions" >/dev/null
    [[ -f "${CHAIN_DIR}/subject_config" && -f "${CHAIN_DIR}/cluster_config" ]] \
        || dmri_fail "chain immutable inputs are missing"
    dmri_require_safe_chain_target "${CHAIN_DIR}/subject_config"
    dmri_require_safe_chain_target "${CHAIN_DIR}/cluster_config"
    IFS= read -r stored_subject < "${CHAIN_DIR}/subject_config" || true
    IFS= read -r stored_cluster < "${CHAIN_DIR}/cluster_config" || true
    [[ "${stored_subject}" == "${subject_config}" ]] \
        || dmri_fail "subject configuration does not match immutable chain input"
    [[ "${stored_cluster}" == "${cluster_config}" ]] \
        || dmri_fail "cluster configuration does not match immutable chain input"
}

DMRI_OWNED_SUBMISSION_LOCK=""

dmri_release_owned_submission_lock() {
    if [[ -n "${DMRI_OWNED_SUBMISSION_LOCK}" ]]; then
        rmdir -- "${DMRI_OWNED_SUBMISSION_LOCK}" 2>/dev/null || true
        DMRI_OWNED_SUBMISSION_LOCK=""
    fi
}

dmri_advance_chain() {
    local successor=$1 subject_config=$2 cluster_config=$3 chain_id=$4
    local marker="${CHAIN_DIR}/${successor}.submitted"
    local lock="${CHAIN_DIR}/.${successor}.submission.lock"
    local submit_status
    if [[ -f "${marker}" ]]; then
        dmri_write_record "${CHAIN_DIR}/status" "${successor}_submitted"
        return 0
    fi
    if ! mkdir -- "${lock}" 2>/dev/null; then
        dmri_fail "successor submission is locked: ${successor}"
    fi
    DMRI_OWNED_SUBMISSION_LOCK=${lock}
    trap dmri_release_owned_submission_lock EXIT
    if [[ -f "${marker}" ]]; then
        dmri_write_record "${CHAIN_DIR}/status" "${successor}_submitted"
        dmri_release_owned_submission_lock
        trap - EXIT
        return 0
    fi
    if dmri_submit_group \
        "${successor}" "${subject_config}" "${cluster_config}" "${chain_id}" "${CHAIN_DIR}"
    then
        dmri_write_record "${CHAIN_DIR}/status" "${successor}_submitted"
        dmri_release_owned_submission_lock
        trap - EXIT
        return 0
    else
        submit_status=$?
    fi
    dmri_write_record "${CHAIN_DIR}/status" "submission_failed"
    dmri_write_record "${CHAIN_DIR}/submission_failed" "${successor}"
    dmri_release_owned_submission_lock
    trap - EXIT
    return "${submit_status}"
}

dmri_run_worker() {
    local group=$1 successor=$2 subject_config=$3 cluster_config=$4 chain_id=$5
    local pipeline_status submit_status
    dmri_load_chain "${subject_config}" "${cluster_config}" "${chain_id}"
    export DMRI_SOFTWARE_CONFIG
    dmri_write_record "${CHAIN_DIR}/${group}.started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if "${REPO_ROOT}/run_pipeline.sh" --stage-group "${group}" "${subject_config}"
    then
        pipeline_status=0
    else
        pipeline_status=$?
    fi
    dmri_write_record "${CHAIN_DIR}/${group}.exit_status" "${pipeline_status}"
    if (( pipeline_status != 0 )); then
        dmri_write_record "${CHAIN_DIR}/${group}.status" "failed"
        dmri_write_record "${CHAIN_DIR}/status" "${group}_failed"
        return "${pipeline_status}"
    fi
    dmri_write_record "${CHAIN_DIR}/${group}.status" "complete"
    dmri_write_record "${CHAIN_DIR}/${group}.completed_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if [[ -z "${successor}" ]]; then
        dmri_write_record "${CHAIN_DIR}/status" "complete"
        dmri_write_record "${CHAIN_DIR}/completed_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        return 0
    fi
    if dmri_advance_chain \
        "${successor}" "${subject_config}" "${cluster_config}" "${chain_id}"
    then
        return 0
    else
        submit_status=$?
    fi
    return "${submit_status}"
}
