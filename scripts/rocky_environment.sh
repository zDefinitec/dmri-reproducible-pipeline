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
    # shellcheck disable=SC1090
    source "${config_path}"
    local name
    for name in CONDA_EXE FSLDIR MATLAB_EXECUTABLE \
        DMRI_EXPECTED_FSL_VERSION DMRI_EXPECTED_MATLAB_VERSION
    do
        [[ -n "${!name:-}" ]] || fail "software configuration is missing ${name}"
        export "${name}"
    done
}

os_release_value() {
    local key=$1 file=$2 value
    value=$(sed -n "s/^${key}=//p" "${file}" | head -n 1)
    value=${value#\"}
    value=${value%\"}
    printf '%s\n' "${value}"
}

check_rocky_platform() {
    local release_file=${DMRI_OS_RELEASE_FILE:-/etc/os-release}
    local os_name architecture os_id version_id
    [[ -f "${release_file}" ]] || fail "cannot read OS release file: ${release_file}"
    os_name=$(uname -s)
    architecture=$(uname -m)
    os_id=$(os_release_value ID "${release_file}")
    version_id=$(os_release_value VERSION_ID "${release_file}")
    [[ "${os_name}" == "Linux" && "${os_id}" == "rocky" \
        && "${version_id}" == "9.7" ]] \
        || fail "Rocky Linux 9.7 is required"
    [[ "${architecture}" == "x86_64" ]] \
        || fail "x86_64 is required; found ${architecture}"
    echo "OK: Rocky Linux ${version_id} ${architecture}"
}
