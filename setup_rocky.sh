#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ENVIRONMENT_FILE="${SCRIPT_DIR}/environment.yml"
ENVIRONMENT_NAME="dmri-repro"
EXPECTED_ATLAS_SHA256="974a0fd72d1214a29e58ccf33cf5aec989d937d999ae65f389dd6b3e1ffdbbad"
EXPECTED_XML_SHA256="2d76ce80d1b0a50dccda2698d5eec55c8984a7f1bb438f79111d67a26fc4dc1c"
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_CACHE_DIR=1

fail() {
    echo "FAIL: $*" >&2
    exit 30
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/rocky_environment.sh"
check_rocky_platform
load_software_config

find_conda() {
    [[ "${CONDA_EXE}" == /* && -x "${CONDA_EXE}" ]] \
        || fail "configured CONDA_EXE must be an absolute executable path"
    printf '%s\n' "${CONDA_EXE}"
}

find_fsldir() {
    [[ "${FSLDIR}" == /* ]] || fail "configured FSLDIR must be an absolute path"
    printf '%s\n' "${FSLDIR}"
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        fail "shasum or sha256sum is required for resource validation"
    fi
}

check_fsl() {
    local root command_name fsl_version
    root=$(find_fsldir)
    [[ -d "${root}" ]] || fail "FSLDIR is not a directory: ${root}"
    for command_name in \
        topup applytopup bet fslmaths eddy_quad flirt fnirt invwarp applywarp
    do
        [[ -x "${root}/bin/${command_name}" ]] \
            || fail "missing required FSL executable: ${command_name}"
    done
    if [[ -x "${root}/bin/eddy_openmp" ]]; then
        echo "OK: FSL EDDY eddy_openmp"
    elif [[ -x "${root}/bin/eddy" ]]; then
        echo "OK: FSL EDDY eddy"
    else
        fail "missing required FSL EDDY executable: eddy_openmp or eddy"
    fi
    for relative in \
        etc/flirtsch/b02b0.cnf \
        etc/flirtsch/b02b0_1.cnf \
        etc/flirtsch/FA_2_FMRIB58_1mm.cnf \
        data/standard/FMRIB58_FA_1mm.nii.gz
    do
        [[ -f "${root}/${relative}" ]] \
            || fail "missing required FSL file: ${relative}"
    done
    [[ -f "${root}/etc/fslversion" ]] \
        || fail "missing required FSL version file: etc/fslversion"
    fsl_version=$(head -n 1 "${root}/etc/fslversion")
    [[ "${fsl_version}" == "${DMRI_EXPECTED_FSL_VERSION}" ]] \
        || fail "FSL version mismatch: expected ${DMRI_EXPECTED_FSL_VERSION}, found ${fsl_version}"
    echo "OK: FSL tools, configs, and standard FA image"
}

check_matlab() {
    local matlab_bin probe output matlab_version matlab_mexext
    matlab_bin=${MATLAB_EXECUTABLE}
    [[ "${matlab_bin}" == /* && -x "${matlab_bin}" ]] \
        || fail "configured MATLAB_EXECUTABLE must be an absolute executable path"
    probe="fprintf('__DMRI_MATLAB_VERSION__%s\\n',version);"
    probe+="fprintf('__DMRI_MEXEXT__%s\\n',mexext);"
    probe+="v=ver('optim'); fprintf('__DMRI_OPT_INSTALLED__%d\\n',~isempty(v));"
    probe+="fprintf('__DMRI_OPT_LICENSED__%d\\n',license('test','Optimization_Toolbox'));"
    probe+="c=mex.getCompilerConfigurations('C','Selected');"
    probe+="fprintf('__DMRI_MEX_CONFIGURED__%d\\n',~isempty(c));"
    probe+="d=tempname; mkdir(d); cleanup_dir=onCleanup(@()rmdir(d,'s'));"
    probe+="src=fullfile(d,'dmri_mex_probe.c'); q=char(34);"
    probe+="code=['#include ' q 'mex.h' q newline "
    probe+="'void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){plhs[0]=mxCreateDoubleScalar(42.0);}' newline];"
    probe+="fid=fopen(src,'w'); assert(fid>=0); fwrite(fid,code); fclose(fid);"
    probe+="mex_works=false; try, mex('-silent','-outdir',d,src);"
    probe+="addpath(d); cleanup_path=onCleanup(@()rmpath(d));"
    probe+="clear dmri_mex_probe; y=dmri_mex_probe();"
    probe+="mex_works=isscalar(y)&&isfinite(y)&&y==42;"
    probe+="clear dmri_mex_probe cleanup_path;"
    probe+="catch ME, disp(getReport(ME,'extended','hyperlinks','off'));"
    probe+="clear dmri_mex_probe; if exist('cleanup_path','var'), clear cleanup_path; end; end;"
    probe+="fprintf('__DMRI_MEX_WORKS__%d\\n',mex_works); clear cleanup_dir;"
    output=$("${matlab_bin}" -batch "${probe}" 2>&1) \
        || fail "MATLAB capability probe failed"
    grep -q '__DMRI_MATLAB_VERSION__.' <<<"${output}" \
        || fail "MATLAB version probe was empty"
    grep -q '__DMRI_MEXEXT__mexa64' <<<"${output}" \
        || fail "MATLAB mexext must be mexa64"
    grep -q '__DMRI_OPT_INSTALLED__1' <<<"${output}" \
        || fail "MATLAB Optimization Toolbox is not installed"
    grep -q '__DMRI_OPT_LICENSED__1' <<<"${output}" \
        || fail "MATLAB Optimization Toolbox is not licensed"
    grep -q '__DMRI_MEX_CONFIGURED__1' <<<"${output}" \
        || fail "MATLAB C MEX compiler is not configured; run mex -setup C"
    grep -q '__DMRI_MEX_WORKS__1' <<<"${output}" \
        || fail "MATLAB C MEX compiler could not compile, load, and run a temporary probe"
    matlab_version=$(sed -n 's/^__DMRI_MATLAB_VERSION__//p' <<<"${output}" | head -n 1)
    matlab_mexext=$(sed -n 's/^__DMRI_MEXEXT__//p' <<<"${output}" | head -n 1)
    [[ "${matlab_version}" == "${DMRI_EXPECTED_MATLAB_VERSION}" ]] \
        || fail "MATLAB version mismatch: expected ${DMRI_EXPECTED_MATLAB_VERSION}, found ${matlab_version}"
    echo "OK: MATLAB ${matlab_version}, ${matlab_mexext}, Optimization Toolbox, and working C MEX compiler"
}

check_python() {
    local conda_bin
    conda_bin=$(find_conda)
    "${conda_bin}" run -n "${ENVIRONMENT_NAME}" python -c \
        "import h5py, matplotlib, nibabel, numpy, pandas, scipy, yaml, dipy; \
expected={'numpy':'2.4.6','scipy':'1.17.1','nibabel':'5.4.2','dipy':'1.12.1',\
'matplotlib':'3.11.1','pandas':'3.0.3','yaml':'6.0.2','h5py':'3.16.0'}; \
modules={'numpy':numpy,'scipy':scipy,'nibabel':nibabel,'dipy':dipy,\
'matplotlib':matplotlib,'pandas':pandas,'yaml':yaml,'h5py':h5py}; \
actual={name: module.__version__ for name,module in modules.items()}; \
assert actual == expected, f'version mismatch: expected={expected}, actual={actual}'; \
print('OK: pinned Python imports and versions', actual)"
}

check_resources() {
    local atlas xml atlas_hash xml_hash
    atlas="${SCRIPT_DIR}/resources/jhu_48roi/JHU-ICBM-labels-2mm.nii.gz"
    xml="${SCRIPT_DIR}/resources/jhu_48roi/JHU-labels.xml"
    [[ -f "${atlas}" && -f "${xml}" ]] || fail "packaged JHU resources are missing"
    atlas_hash=$(sha256_file "${atlas}")
    xml_hash=$(sha256_file "${xml}")
    [[ "${atlas_hash}" == "${EXPECTED_ATLAS_SHA256}" ]] \
        || fail "packaged JHU atlas image hash mismatch"
    [[ "${xml_hash}" == "${EXPECTED_XML_SHA256}" ]] \
        || fail "packaged JHU atlas XML hash mismatch"
    echo "OK: packaged JHU image/XML hashes"
}

check_disk() {
    local available_kib
    available_kib=$(df -Pk "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')
    [[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -gt 0 ]] \
        || fail "could not determine available free disk space"
    echo "OK: available disk $((available_kib / 1024)) MiB"
}

run_checks() {
    find_conda >/dev/null
    echo "OK: Conda/Miniforge"
    check_fsl
    check_matlab
    check_python
    check_resources
    check_disk
}

usage() {
    echo "Usage: ./setup_rocky.sh [--check]" >&2
}

case "${1:-}" in
    "")
        conda_bin=$(find_conda)
        "${conda_bin}" env update -n "${ENVIRONMENT_NAME}" \
            -f "${ENVIRONMENT_FILE}" --prune
        (
            cd "${SCRIPT_DIR}"
            "${conda_bin}" run -n "${ENVIRONMENT_NAME}" python -m pip install \
                --no-deps --no-build-isolation .
        )
        run_checks
        ;;
    --check)
        [[ "$#" -eq 1 ]] || { usage; exit 2; }
        run_checks
        ;;
    *)
        usage
        exit 2
        ;;
esac
