#!/usr/bin/env bash

load_software_config() {
    local config_path=${DMRI_SOFTWARE_CONFIG:-}
    [[ -n "${config_path}" ]] \
        || fail "DMRI_SOFTWARE_CONFIG must name the private server software configuration"
    case "${config_path}" in
        /*) ;;
        *) fail "DMRI_SOFTWARE_CONFIG must be an absolute path" ;;
    esac
    [[ -f "${config_path}" && ! -L "${config_path}" && -r "${config_path}" ]] \
        || fail "DMRI_SOFTWARE_CONFIG must be a readable regular file, not a symlink"
    [[ "$(stat -c '%u' "${config_path}")" == "$(id -u)" ]] \
        || fail "DMRI_SOFTWARE_CONFIG must be owned by the current user"
    local mode
    mode=$(stat -c '%a' "${config_path}")
    (( (8#${mode} & 8#022) == 0 )) \
        || fail "DMRI_SOFTWARE_CONFIG must not be group- or world-writable"
    local payload
    if ! payload=$(
        /usr/bin/env -i /bin/bash --noprofile --norc -c '
            set -euo pipefail
            config_path=$1
            required=(
                CONDA_EXE FSLDIR MATLAB_EXECUTABLE
                DMRI_EXPECTED_FSL_VERSION DMRI_EXPECTED_MATLAB_VERSION
            )
            unset "${required[@]}"
            # Config diagnostics remain diagnostics; stdout is reserved for
            # the fixed, validated import protocol below.
            # shellcheck disable=SC1090
            source "${config_path}" 1>&2
            for name in "${required[@]}"; do
                builtin printf "%s:" "${name}"
                builtin printf "%s" "${!name-}" \
                    | /usr/bin/od -An -v -tx1 \
                    | /usr/bin/tr -d " \\n"
                builtin printf "\\n"
            done
        ' dmri-software-config "${config_path}"
    ); then
        fail "software configuration could not be loaded"
    fi

    local required=(
        CONDA_EXE FSLDIR MATLAB_EXECUTABLE
        DMRI_EXPECTED_FSL_VERSION DMRI_EXPECTED_MATLAB_VERSION
    )
    local index=0 line name encoded escapes value offset
    while IFS= read -r line; do
        [[ "${index}" -lt "${#required[@]}" ]] \
            || fail "software configuration returned malformed values"
        name=${line%%:*}
        encoded=${line#*:}
        [[ "${name}" == "${required[index]}" && "${line}" == *:* ]] \
            || fail "software configuration returned malformed values"
        [[ "${encoded}" =~ ^([0-9a-f][0-9a-f])*$ ]] \
            || fail "software configuration returned malformed values"
        escapes=
        for ((offset = 0; offset < ${#encoded}; offset += 2)); do
            escapes+="\\x${encoded:offset:2}"
        done
        value=
        if [[ -n "${encoded}" ]]; then
            printf -v value '%b' "${escapes}"
        fi
        [[ -n "${value}" ]] \
            || fail "software configuration is missing ${name}"
        export "${name}=${value}"
        ((index += 1))
    done <<<"${payload}"
    [[ "${index}" -eq "${#required[@]}" ]] \
        || fail "software configuration returned malformed values"
}

os_release_value() {
    local key=$1 file=$2 value
    value=$(sed -n "s/^${key}=//p" "${file}" | head -n 1)
    value=${value#\"}
    value=${value%\"}
    printf '%s\n' "${value}"
}

check_rocky_platform() {
    local release_file=${1:-/etc/os-release}
    local uname_bin=${2:-/usr/bin/uname}
    local os_name architecture distribution_name os_id version_id
    case "${release_file}" in
        /*) ;;
        *) fail "OS release file must be an absolute path" ;;
    esac
    case "${uname_bin}" in
        /*) ;;
        *) fail "uname command must be an absolute path" ;;
    esac
    [[ -f "${release_file}" ]] || fail "cannot read OS release file: ${release_file}"
    [[ -f "${uname_bin}" && -x "${uname_bin}" ]] \
        || fail "cannot execute uname command: ${uname_bin}"
    os_name=$("${uname_bin}" -s)
    architecture=$("${uname_bin}" -m)
    distribution_name=$(os_release_value NAME "${release_file}")
    os_id=$(os_release_value ID "${release_file}")
    version_id=$(os_release_value VERSION_ID "${release_file}")
    [[ "${os_name}" == "Linux" && "${distribution_name}" == "Rocky Linux" \
        && "${os_id}" == "rocky" \
        && "${version_id}" == "9.7" ]] \
        || fail "Rocky Linux 9.7 is required"
    [[ "${architecture}" == "x86_64" ]] \
        || fail "x86_64 is required; found ${architecture}"
    echo "OK: Rocky Linux ${version_id} ${architecture}"
}
