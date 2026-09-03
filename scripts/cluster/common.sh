#!/usr/bin/env bash
set -euo pipefail

CLUSTER_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(CDPATH= cd -- "${CLUSTER_SCRIPT_DIR}/../.." && pwd -P)

dmri_fail() {
    echo "ERROR: $*" >&2
    exit 30
}

fail() {
    dmri_fail "$@"
}

# Reuse the public wrapper's private software-config loader and its fixed-key,
# clean-environment import protocol.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/rocky_environment.sh"

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
    load_private_shell_config_values \
        "${cluster_config}" \
        "cluster configuration" \
        "cluster configuration" \
        "${required_keys[@]}"
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
    load_software_config
    configured_conda=${CONDA_EXE}
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
    dmri_require_record_target "${target}"
    if [[ -e "${temporary}" || -L "${temporary}" ]]; then
        dmri_fail "temporary record path already exists: ${temporary}"
    fi
    if ! printf '%s\n' "${value}" > "${temporary}"; then
        dmri_fail "could not write temporary record: ${temporary}"
    fi
    [[ -f "${temporary}" && ! -L "${temporary}" ]] \
        || dmri_fail "temporary record is not a regular file: ${temporary}"
    if ! mv -f -- "${temporary}" "${target}"; then
        dmri_fail "could not publish record: ${target}"
    fi
    [[ -f "${target}" && ! -L "${target}" ]] \
        || dmri_fail "published record is not a regular file: ${target}"
}

dmri_require_record_target() {
    local target=$1
    if [[ -L "${target}" || ( -e "${target}" && ! -f "${target}" ) ]]; then
        dmri_fail "record target must be absent or a regular non-symlink file: ${target}"
    fi
}

dmri_write_arguments_record() {
    local target=$1 argument quoted contents=""
    shift
    for argument in "$@"; do
        if ! printf -v quoted '%q' "${argument}"; then
            dmri_fail "could not encode submission argument record: ${target}"
        fi
        contents="${contents}${quoted}"$'\n'
    done
    contents=${contents%$'\n'}
    dmri_write_record "${target}" "${contents}"
}

dmri_prepare_capture_record() {
    local target=$1 temporary=$2
    dmri_require_record_target "${target}"
    if [[ -e "${temporary}" || -L "${temporary}" ]]; then
        dmri_fail "temporary record path already exists: ${temporary}"
    fi
    if ! : > "${temporary}"; then
        dmri_fail "could not create temporary record: ${temporary}"
    fi
    [[ -f "${temporary}" && ! -L "${temporary}" ]] \
        || dmri_fail "temporary record is not a regular file: ${temporary}"
}

dmri_publish_capture_record() {
    local temporary=$1 target=$2
    dmri_require_record_target "${target}"
    [[ -f "${temporary}" && ! -L "${temporary}" ]] \
        || dmri_fail "temporary record is not a regular file: ${temporary}"
    if ! mv -f -- "${temporary}" "${target}"; then
        dmri_fail "could not publish record: ${target}"
    fi
    [[ -f "${target}" && ! -L "${target}" ]] \
        || dmri_fail "published record is not a regular file: ${target}"
}

dmri_require_safe_chain_target() {
    local target=$1
    [[ ! -L "${target}" ]] \
        || dmri_fail "chain path must resolve below CLUSTER_RUN_ROOT without symbolic links: ${target}"
}

dmri_submit_group() {
    local group=$1 subject_config=$2 cluster_config=$3 chain_id=$4 chain_dir=$5
    local wrapper command status submitted_at
    local argv_record stdout_record stderr_record exit_record marker
    local stdout_temporary stderr_temporary
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
    argv_record="${chain_dir}/submissions/${group}.argv"
    stdout_record="${chain_dir}/submissions/${group}.stdout"
    stderr_record="${chain_dir}/submissions/${group}.stderr"
    exit_record="${chain_dir}/submissions/${group}.exit_status"
    marker="${chain_dir}/${group}.submitted"
    stdout_temporary="${stdout_record}.tmp.$$"
    stderr_temporary="${stderr_record}.tmp.$$"
    dmri_require_safe_chain_target "${argv_record}"
    dmri_require_safe_chain_target "${stdout_record}"
    dmri_require_safe_chain_target "${stderr_record}"
    dmri_require_safe_chain_target "${exit_record}"
    dmri_require_safe_chain_target "${marker}"
    dmri_require_record_target "${argv_record}"
    dmri_require_record_target "${stdout_record}"
    dmri_require_record_target "${stderr_record}"
    dmri_require_record_target "${exit_record}"
    dmri_require_record_target "${marker}"
    dmri_write_arguments_record "${argv_record}" "${submit_arguments[@]}"
    dmri_prepare_capture_record "${stdout_record}" "${stdout_temporary}"
    dmri_prepare_capture_record "${stderr_record}" "${stderr_temporary}"
    if "${CBIG_PBSUBMIT}" "${submit_arguments[@]}" \
        > "${stdout_temporary}" \
        2> "${stderr_temporary}"
    then
        status=0
    else
        status=$?
    fi
    if (( status == 0 )); then
        DMRI_RETAIN_OWNED_SUBMISSION_LOCK=1
        if ! submitted_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ'); then
            dmri_fail "could not timestamp accepted ${group} submission"
        fi
        dmri_write_record "${marker}" "${submitted_at}"
        DMRI_RETAIN_OWNED_SUBMISSION_LOCK=0
    fi
    dmri_publish_capture_record "${stdout_temporary}" "${stdout_record}"
    dmri_publish_capture_record "${stderr_temporary}" "${stderr_record}"
    dmri_write_record "${exit_record}" "${status}"
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
    local subject_config=$1 cluster_config=$2 chain_id=$3
    local stored_subject stored_cluster stored_subject_id stored_subject_output
    local immutable_noddi_workers
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
    [[ -f "${CHAIN_DIR}/subject_config" \
        && -f "${CHAIN_DIR}/cluster_config" \
        && -f "${CHAIN_DIR}/subject_id" \
        && -f "${CHAIN_DIR}/subject_output" \
        && -f "${CHAIN_DIR}/noddi_workers" ]] \
        || dmri_fail "chain immutable inputs are missing"
    dmri_require_safe_chain_target "${CHAIN_DIR}/subject_config"
    dmri_require_safe_chain_target "${CHAIN_DIR}/cluster_config"
    dmri_require_safe_chain_target "${CHAIN_DIR}/subject_id"
    dmri_require_safe_chain_target "${CHAIN_DIR}/subject_output"
    dmri_require_safe_chain_target "${CHAIN_DIR}/noddi_workers"
    stored_subject=$(< "${CHAIN_DIR}/subject_config")
    stored_cluster=$(< "${CHAIN_DIR}/cluster_config")
    stored_subject_id=$(< "${CHAIN_DIR}/subject_id")
    stored_subject_output=$(< "${CHAIN_DIR}/subject_output")
    dmri_require_value "immutable subject configuration" "${stored_subject}"
    dmri_require_value "immutable cluster configuration" "${stored_cluster}"
    dmri_require_value "immutable subject_id" "${stored_subject_id}"
    dmri_require_value "immutable subject_output" "${stored_subject_output}"
    [[ "${stored_subject_output}" == /* ]] \
        || dmri_fail "immutable subject_output must be an absolute path"
    [[ "${stored_subject}" == "${subject_config}" ]] \
        || dmri_fail "subject configuration does not match immutable chain input"
    [[ "${stored_cluster}" == "${cluster_config}" ]] \
        || dmri_fail "cluster configuration does not match immutable chain input"
    immutable_noddi_workers=$(< "${CHAIN_DIR}/noddi_workers")
    dmri_validate_positive_integer \
        "immutable noddi_workers" "${immutable_noddi_workers}"
    if (( NODDI_NCPUS < immutable_noddi_workers )); then
        dmri_fail "NODDI_NCPUS must be at least the immutable noddi_workers"
    fi
    dmri_read_subject_context "${subject_config}"
    [[ "${SUBJECT_ID}" == "${stored_subject_id}" ]] \
        || dmri_fail "live subject_id does not match immutable chain context"
    [[ "${SUBJECT_OUTPUT}" == "${stored_subject_output}" ]] \
        || dmri_fail "live subject_output does not match immutable chain context"
    [[ "${NODDI_WORKERS}" == "${immutable_noddi_workers}" ]] \
        || dmri_fail "live noddi_workers does not match immutable chain context"
}

DMRI_OWNED_SUBMISSION_LOCK=""
DMRI_RETAIN_OWNED_SUBMISSION_LOCK=0

dmri_release_owned_submission_lock() {
    local owner
    [[ -n "${DMRI_OWNED_SUBMISSION_LOCK}" ]] || return 0
    (( DMRI_RETAIN_OWNED_SUBMISSION_LOCK == 0 )) || return 0
    owner="${DMRI_OWNED_SUBMISSION_LOCK}/owner"
    if [[ -L "${owner}" || ( -e "${owner}" && ! -f "${owner}" ) ]]; then
        return 30
    fi
    if [[ -f "${owner}" ]] && ! rm -f -- "${owner}"; then
        return 30
    fi
    if ! rmdir -- "${DMRI_OWNED_SUBMISSION_LOCK}" 2>/dev/null; then
        return 30
    fi
    DMRI_OWNED_SUBMISSION_LOCK=""
    return 0
}

dmri_advance_chain() {
    local source_group=$1 successor=$2 subject_config=$3 cluster_config=$4 chain_id=$5
    local marker="${CHAIN_DIR}/${successor}.submitted"
    local lock="${CHAIN_DIR}/.${successor}.submission.lock"
    local submit_status acquired_at lock_owner
    dmri_require_record_target "${marker}"
    if [[ -f "${marker}" ]]; then
        return 0
    fi
    dmri_require_record_target "${CHAIN_DIR}/status"
    dmri_require_record_target "${CHAIN_DIR}/submission_failed"
    if ! mkdir -- "${lock}" 2>/dev/null; then
        dmri_fail "successor submission is locked: ${successor}"
    fi
    DMRI_OWNED_SUBMISSION_LOCK=${lock}
    DMRI_RETAIN_OWNED_SUBMISSION_LOCK=0
    trap 'dmri_release_owned_submission_lock || true' EXIT
    if ! acquired_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ'); then
        dmri_fail "could not timestamp successor submission lock"
    fi
    lock_owner="chain_id=${chain_id}"$'\n'
    lock_owner="${lock_owner}source_group=${source_group}"$'\n'
    lock_owner="${lock_owner}successor=${successor}"$'\n'
    lock_owner="${lock_owner}job_name=dmri_${successor}_${chain_id}"$'\n'
    lock_owner="${lock_owner}pid=$$"$'\n'
    lock_owner="${lock_owner}acquired_utc=${acquired_at}"
    dmri_write_record "${lock}/owner" "${lock_owner}"
    dmri_require_record_target "${marker}"
    if [[ -f "${marker}" ]]; then
        dmri_release_owned_submission_lock \
            || dmri_fail "could not release owned successor submission lock"
        trap - EXIT
        return 0
    fi
    if dmri_submit_group \
        "${successor}" "${subject_config}" "${cluster_config}" "${chain_id}" "${CHAIN_DIR}"
    then
        dmri_write_record "${CHAIN_DIR}/status" "${successor}_submitted"
        dmri_release_owned_submission_lock \
            || dmri_fail "could not release owned successor submission lock"
        trap - EXIT
        return 0
    else
        submit_status=$?
    fi
    dmri_write_record "${CHAIN_DIR}/status" "submission_failed"
    dmri_write_record "${CHAIN_DIR}/submission_failed" "${successor}"
    dmri_release_owned_submission_lock \
        || dmri_fail "could not release owned successor submission lock"
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
        "${group}" "${successor}" "${subject_config}" "${cluster_config}" "${chain_id}"
    then
        return 0
    else
        submit_status=$?
    fi
    return "${submit_status}"
}
