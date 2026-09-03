#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

start_group=topup
if [[ "${1-}" == "--start-at" ]]; then
    [[ $# -eq 4 ]] || dmri_fail "usage: $0 [--start-at topup|eddy|noddi] SUBJECT.yaml CLUSTER.local.sh"
    start_group=$2
    shift 2
fi
[[ $# -eq 2 ]] || dmri_fail "usage: $0 [--start-at topup|eddy|noddi] SUBJECT.yaml CLUSTER.local.sh"
case "${start_group}" in
    topup|eddy|noddi) ;;
    *) dmri_fail "--start-at must be topup, eddy, or noddi" ;;
esac

subject_config=$1
cluster_config=$2
dmri_validate_input_paths "${subject_config}" "${cluster_config}"
dmri_load_cluster_config "${cluster_config}"
dmri_read_subject_context "${subject_config}"
if (( NODDI_NCPUS < NODDI_WORKERS )); then
    dmri_fail "NODDI_NCPUS must be at least the configured noddi_workers"
fi

mkdir -p -- "${CLUSTER_RUN_ROOT}"
CLUSTER_RUN_ROOT=$(CDPATH= cd -- "${CLUSTER_RUN_ROOT}" && pwd -P)
dmri_acquire_subject_submission_lock "${start_group}"
attempt=0
while :; do
    attempt=$((attempt + 1))
    chain_id="chain-$(date -u '+%Y%m%dT%H%M%SZ')-$$-${RANDOM}-${attempt}"
    chain_dir="${CLUSTER_RUN_ROOT}/${chain_id}"
    if mkdir -- "${chain_dir}" 2>/dev/null; then
        break
    fi
    (( attempt < 100 )) || dmri_fail "could not allocate a unique chain directory"
done
mkdir -- "${chain_dir}/logs" "${chain_dir}/submissions"
dmri_write_record "${chain_dir}/chain_id" "${chain_id}"
dmri_write_record "${chain_dir}/subject_config" "${subject_config}"
dmri_write_record "${chain_dir}/cluster_config" "${cluster_config}"
dmri_write_record "${chain_dir}/subject_id" "${SUBJECT_ID}"
dmri_write_record "${chain_dir}/subject_output" "${SUBJECT_OUTPUT}"
dmri_write_record "${chain_dir}/noddi_workers" "${NODDI_WORKERS}"
dmri_write_record "${chain_dir}/start_group" "${start_group}"
dmri_write_record "${chain_dir}/created_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
chmod 0444 \
    "${chain_dir}/chain_id" \
    "${chain_dir}/subject_config" \
    "${chain_dir}/cluster_config" \
    "${chain_dir}/subject_id" \
    "${chain_dir}/subject_output" \
    "${chain_dir}/noddi_workers" \
    "${chain_dir}/start_group" \
    "${chain_dir}/created_at"
dmri_record_subject_submission_attempt "${chain_id}"
dmri_require_record_target "${chain_dir}/status"

if dmri_submit_group \
    "${start_group}" "${subject_config}" "${cluster_config}" "${chain_id}" "${chain_dir}"
then
    dmri_write_record "${chain_dir}/status" "submitted"
    if ! printf 'Submitted chain %s at %s\n' "${chain_id}" "${chain_dir}"; then
        dmri_fail "could not report accepted chain submission"
    fi
    DMRI_RETAIN_OWNED_SUBMISSION_LOCK=0
    dmri_release_owned_submission_lock \
        || dmri_fail "could not release owned subject submission lock"
    trap - EXIT
else
    submit_status=$?
    dmri_write_record "${chain_dir}/status" "submission_failed"
    DMRI_RETAIN_OWNED_SUBMISSION_LOCK=0
    dmri_release_owned_submission_lock \
        || dmri_fail "could not release owned subject submission lock"
    trap - EXIT
    exit "${submit_status}"
fi
exit 0
