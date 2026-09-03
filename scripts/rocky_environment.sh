#!/usr/bin/env bash

load_private_shell_config_values() {
    local config_path=$1 path_label=$2 config_label=$3
    shift 3
    local required=("$@")
    [[ -n "${config_path}" ]] \
        || fail "${path_label} must name a private shell configuration"
    case "${config_path}" in
        /*) ;;
        *) fail "${path_label} must be an absolute path" ;;
    esac
    [[ -f "${config_path}" && ! -L "${config_path}" && -r "${config_path}" ]] \
        || fail "${path_label} must be a readable regular file, not a symlink"
    [[ "$(stat -c '%u' "${config_path}")" == "$(id -u)" ]] \
        || fail "${path_label} must be owned by the current user"
    local mode
    mode=$(stat -c '%a' "${config_path}")
    [[ "${mode}" =~ ^[0-7][0-7][0-7][0-7]?$ ]] \
        || fail "${path_label} returned invalid permission metadata"
    (( (8#${mode} & 8#022) == 0 )) \
        || fail "${path_label} must not be group- or world-writable"
    local required_name
    (( ${#required[@]} > 0 )) \
        || fail "${config_label} has no fixed import keys"
    for required_name in "${required[@]}"; do
        [[ "${required_name}" =~ ^[A-Z][A-Z0-9_]*$ ]] \
            || fail "${config_label} has an invalid import key"
    done
    unset "${required[@]}" \
        || fail "${config_label} fixed keys could not be reset before import"
    local payload
    if ! payload=$(
        /usr/bin/env -i /bin/bash --noprofile --norc -c '
            set -euo pipefail
            config_path=$1
            shift
            required=("$@")
            unset "${required[@]}"
            readonly config_path
            readonly -a required
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
        ' dmri-private-config "${config_path}" "${required[@]}"
    ); then
        fail "${config_label} could not be loaded"
    fi

    local index=0 line name encoded escapes value offset
    while IFS= read -r line; do
        [[ "${index}" -lt "${#required[@]}" ]] \
            || fail "${config_label} returned malformed values"
        name=${line%%:*}
        encoded=${line#*:}
        [[ "${name}" == "${required[index]}" && "${line}" == *:* ]] \
            || fail "${config_label} returned malformed values"
        [[ "${encoded}" =~ ^([0-9a-f][0-9a-f])*$ ]] \
            || fail "${config_label} returned malformed values"
        escapes=
        for ((offset = 0; offset < ${#encoded}; offset += 2)); do
            escapes+="\\x${encoded:offset:2}"
        done
        value=
        if [[ -n "${encoded}" ]]; then
            printf -v value '%b' "${escapes}"
        fi
        [[ -n "${value}" ]] \
            || fail "${config_label} is missing ${name} (required)"
        printf -v "${name}" '%s' "${value}"
        ((index += 1))
    done <<<"${payload}"
    [[ "${index}" -eq "${#required[@]}" ]] \
        || fail "${config_label} returned malformed values"
}

load_software_config() {
    local config_path=${DMRI_SOFTWARE_CONFIG:-}
    local required=(
        CONDA_EXE FSLDIR MATLAB_EXECUTABLE
        DMRI_EXPECTED_FSL_VERSION DMRI_EXPECTED_MATLAB_VERSION
    )
    load_private_shell_config_values \
        "${config_path}" \
        "DMRI_SOFTWARE_CONFIG" \
        "software configuration" \
        "${required[@]}"
    export "${required[@]}"
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
