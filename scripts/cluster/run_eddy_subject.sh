#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

[[ $# -eq 3 ]] \
    || dmri_fail "usage: $0 SUBJECT.yaml CLUSTER.local.sh CHAIN_ID"
dmri_run_worker eddy noddi "$1" "$2" "$3"
