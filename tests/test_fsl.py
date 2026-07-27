from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmri_pipeline.fsl import (
    ExternalCommandError,
    FSLContext,
    FSLDiscoveryError,
    FSLInstallation,
    build_applytopup_command,
    build_bet_command,
    build_eddy_command,
    build_eddy_quad_command,
    build_jhu_commands,
    build_topup_command,
    build_topup_mean_command,
    discover_fsl,
    run_fsl_command,
)


_TOOLS = (
    "topup",
    "applytopup",
    "bet",
    "fslmaths",
    "eddy",
    "eddy_quad",
    "flirt",
    "fnirt",
    "invwarp",
    "applywarp",
)
_RESOURCES = (
    "etc/flirtsch/b02b0.cnf",
    "etc/flirtsch/b02b0_1.cnf",
    "etc/flirtsch/FA_2_FMRIB58_1mm.cnf",
    "data/standard/FMRIB58_FA_1mm.nii.gz",
)
_EDDY_QUAD_DEPENDENCIES = (
    ("numpy", "2.1.3"),
    ("nibabel", "5.4.2"),
    ("matplotlib", "3.9.4"),
    ("seaborn", "0.13.2"),
)
_PYTHON_SITE_PACKAGES = "lib/python3.12/site-packages"


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fake_fsl(
    root: Path,
    *,
    include_openmp: bool = True,
    omit: str | None = None,
) -> Path:
    def is_omitted(relative: str, *, legacy_name: str | None = None) -> bool:
        if omit is None:
            return False
        return (
            omit == relative
            or omit == legacy_name
            or relative.startswith(omit.rstrip("/") + "/")
        )

    def write_file(
        relative: str,
        body: str | None = None,
        *,
        executable: bool = False,
        legacy_name: str | None = None,
    ) -> None:
        if is_omitted(relative, legacy_name=legacy_name):
            return
        path = root / relative
        if executable:
            if body is None:
                _write_executable(path)
            else:
                _write_executable(path, body)
            return
        content = body if body is not None else f"{relative}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    for tool in _TOOLS:
        body = None
        if tool == "eddy":
            body = (
                "#!/usr/bin/env fslpython\n"
                "from fsl.base import find_cuda_exe\n"
            )
        elif tool == "eddy_quad":
            body = (
                "#!/bin/sh\n"
                "'''exec' \"${FSLDIR}/bin/python\" \"$0\" \"$@\"\n"
                "' '''\n"
                "from eddy_qc.scripts.eddy_quad import main\n"
            )
        write_file(
            f"bin/{tool}",
            body,
            executable=True,
            legacy_name=tool,
        )
    if include_openmp:
        write_file(
            "bin/eddy_openmp",
            executable=True,
            legacy_name="eddy_openmp",
        )
    for helper in (
        "bet2",
        "remove_ext",
        "imtest",
        "imrm",
        "imglob",
        "fslval",
        "fslhd",
        "fslstats",
        "fslsplit",
        "slicer",
    ):
        write_file(
            f"bin/{helper}",
            executable=True,
            legacy_name=helper,
        )
    if not include_openmp:
        for helper in ("eddy_cpu", "find_cuda_exe"):
            write_file(
                f"bin/{helper}",
                executable=True,
                legacy_name=helper,
            )
    write_file(
        "bin/fslpython",
        '#!/bin/sh\nexec "${FSLDIR}/bin/python" "$@"\n',
        executable=True,
    )
    write_file("bin/python3.12", executable=True, legacy_name="python3.12")
    if not is_omitted("bin/python"):
        (root / "bin" / "python").symlink_to("python3.12")

    for relative in (
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/scripts/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/scripts/eddy_quad.py",
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/QUAD/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/QUAD/quad.py",
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/utils/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/eddy_qc/utils/eddy_qc_logos.png",
    ):
        write_file(relative)
    for relative in (
        f"{_PYTHON_SITE_PACKAGES}/fsl/utils/run.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/base/find_cuda_exe.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/scripts/remove_ext.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/scripts/imtest.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/utils/path.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/wrappers/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/data/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/data/image.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/transform/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/transform/affine.py",
        f"{_PYTHON_SITE_PACKAGES}/fsl/base/__init__.py",
    ):
        write_file(relative)
    for relative in (
        f"{_PYTHON_SITE_PACKAGES}/numpy/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/numpy/_core/fromnumeric.py",
        f"{_PYTHON_SITE_PACKAGES}/nibabel/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/nibabel/nifti1.py",
        f"{_PYTHON_SITE_PACKAGES}/matplotlib/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/matplotlib/backends/backend_pdf.py",
        f"{_PYTHON_SITE_PACKAGES}/matplotlib/mpl-data/matplotlibrc",
        f"{_PYTHON_SITE_PACKAGES}/seaborn/__init__.py",
        f"{_PYTHON_SITE_PACKAGES}/seaborn/categorical.py",
    ):
        write_file(relative)
    for distribution, version in (
        ("eddy_qc", "1.4.2"),
        *_EDDY_QUAD_DEPENDENCIES,
    ):
        metadata_root = (
            f"{_PYTHON_SITE_PACKAGES}/"
            f"{distribution}-{version}.dist-info"
        )
        write_file(
            f"{metadata_root}/METADATA",
            (
                "Metadata-Version: 2.1\n"
                f"Name: {distribution}\n"
                f"Version: {version}\n"
            ),
        )
        write_file(f"{metadata_root}/RECORD")
    for resource in _RESOURCES:
        if not is_omitted(resource):
            write_file(resource, "fake resource\n")
    return root


@pytest.fixture
def fake_fsldir(tmp_path: Path) -> Path:
    return _fake_fsl(tmp_path / "fake fsl")


@pytest.fixture
def fsl_installation(fake_fsldir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FSL_TEST_UNRELATED", "preserved")
    return discover_fsl(SimpleNamespace(fsldir=fake_fsldir))


@pytest.fixture
def fsl_context(tmp_path: Path, fsl_installation: FSLInstallation) -> FSLContext:
    work = tmp_path / "subject output"
    raw = tmp_path / "raw inputs"
    atlas = tmp_path / "package resources" / "JHU labels 48.nii.gz"
    return FSLContext(
        installation=fsl_installation,
        merged_b0=work / "03_topup" / "PA_AP_b0.nii.gz",
        merged_b0_shape=(120, 120, 80, 18),
        acqparams_topup=work / "03_topup" / "acqparams_topup.txt",
        topup_prefix=work / "03_topup" / "topup_PA_AP_b0",
        topup_corrected_b0s=work / "03_topup" / "topup_corrected_b0s",
        field_hz_prefix=work / "03_topup" / "topup_field_Hz",
        applytopup_inputs=(
            work / "02_gibbs" / "pa_b0s.nii.gz",
            work / "02_gibbs" / "ap_b0s.nii.gz",
        ),
        applytopup_indices=(1, 2),
        applytopup_output=work / "03_topup" / "applytopup_corrected_b0s",
        hifi_nodif=work / "04_bet" / "hifi_nodif",
        brain_prefix=work / "04_bet" / "hifi_nodif_brain",
        gibbs_pa=work / "02_gibbs" / "dMRI_PA_pca_gibbs.nii.gz",
        cleaned_mask=work / "04_bet" / "hifi_nodif_brain_mask_lcc.nii.gz",
        acqparams_eddy=work / "03_topup" / "acqparams_eddy.txt",
        index_eddy=work / "03_topup" / "index_eddy.txt",
        bvals=raw / "pa_dwi.bval",
        bvecs=raw / "pa_dwi.bvec",
        eddy_prefix=work / "05_eddy" / "eddy_unwarped_images",
        eddy_threads=12,
        eddy_quad_output=work / "05_eddy" / "eddy_quad",
        subject_fa=work / "06_dti" / "FA.nii.gz",
        affine_fa=work / "09_jhu" / "FA_in_standard_affine.nii.gz",
        affine_matrix=work / "09_jhu" / "dti2standard_affine.mat",
        nonlinear_fa=work / "09_jhu" / "FA_in_standard_nonlinear.nii.gz",
        forward_warp=work / "09_jhu" / "dti2standard_warp.nii.gz",
        inverse_warp=work / "09_jhu" / "standard2dti_warp.nii.gz",
        atlas_labels=atlas,
        subject_atlas=work / "09_jhu" / "WM_JHU_ROIs.nii.gz",
    )


def test_discovery_prefers_explicit_fsldir_over_environment_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    explicit = _fake_fsl(tmp_path / "explicit")
    environment = _fake_fsl(tmp_path / "environment")
    monkeypatch.setenv("FSLDIR", str(environment))
    monkeypatch.setenv("PATH", str(environment / "bin"))

    installation = discover_fsl(SimpleNamespace(fsldir=explicit))

    assert installation.fsldir == explicit
    assert installation.topup == explicit / "bin" / "topup"


def test_discovery_rejects_invalid_explicit_fsldir_without_falling_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fallback = _fake_fsl(tmp_path / "fallback")
    monkeypatch.setenv("FSLDIR", str(fallback))
    monkeypatch.setenv("PATH", str(fallback / "bin"))

    with pytest.raises(FSLDiscoveryError, match="explicit"):
        discover_fsl(SimpleNamespace(fsldir=tmp_path / "invalid"))


def test_discovery_uses_process_fsldir_before_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _fake_fsl(tmp_path / "environment")
    path_fsl = _fake_fsl(tmp_path / "path")
    monkeypatch.setenv("FSLDIR", str(environment))
    monkeypatch.setenv("PATH", str(path_fsl / "bin"))

    assert discover_fsl(SimpleNamespace(fsldir=None)).fsldir == environment


def test_discovery_infers_root_from_topup_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path_fsl = _fake_fsl(tmp_path / "path installation")
    monkeypatch.delenv("FSLDIR", raising=False)
    monkeypatch.setenv("PATH", str(path_fsl / "bin"))

    installation = discover_fsl(SimpleNamespace(fsldir=None))

    assert installation.fsldir == path_fsl
    assert installation.applytopup == path_fsl / "bin" / "applytopup"


def test_discovery_prefers_eddy_openmp(fake_fsldir: Path):
    installation = discover_fsl(SimpleNamespace(fsldir=fake_fsldir))
    assert installation.eddy == fake_fsldir / "bin" / "eddy_openmp"


def test_discovery_falls_back_to_cpu_eddy(tmp_path: Path):
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    installation = discover_fsl(SimpleNamespace(fsldir=fsldir))
    assert installation.eddy == fsldir / "bin" / "eddy"


def test_discovery_records_bounded_relative_runtime_material_files(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)

    installation = discover_fsl(SimpleNamespace(fsldir=fsldir))

    material = dict(installation.runtime_material_files)
    assert {
        "bin/bet2",
        "bin/remove_ext",
        "bin/imtest",
        "bin/imrm",
        "bin/imglob",
        "bin/fslval",
        "bin/fslhd",
        "bin/fslstats",
        "bin/fslsplit",
        "bin/slicer",
        "bin/eddy_cpu",
        "bin/find_cuda_exe",
        "bin/fslpython",
        "bin/python3.12",
        "lib/python3.12/site-packages/eddy_qc/QUAD/quad.py",
        "lib/python3.12/site-packages/eddy_qc/scripts/eddy_quad.py",
        "lib/python3.12/site-packages/eddy_qc-1.4.2.dist-info/METADATA",
        "lib/python3.12/site-packages/fsl/scripts/remove_ext.py",
        "lib/python3.12/site-packages/fsl/scripts/imtest.py",
        "lib/python3.12/site-packages/fsl/utils/path.py",
        "lib/python3.12/site-packages/fsl/data/image.py",
        "lib/python3.12/site-packages/fsl/wrappers/__init__.py",
        "lib/python3.12/site-packages/numpy/__init__.py",
        "lib/python3.12/site-packages/numpy/_core/fromnumeric.py",
        "lib/python3.12/site-packages/nibabel/__init__.py",
        "lib/python3.12/site-packages/nibabel/nifti1.py",
        "lib/python3.12/site-packages/matplotlib/__init__.py",
        "lib/python3.12/site-packages/matplotlib/backends/backend_pdf.py",
        "lib/python3.12/site-packages/matplotlib/mpl-data/matplotlibrc",
        "lib/python3.12/site-packages/seaborn/__init__.py",
        "lib/python3.12/site-packages/seaborn/categorical.py",
        "lib/python3.12/site-packages/numpy-2.1.3.dist-info/METADATA",
        "lib/python3.12/site-packages/nibabel-5.4.2.dist-info/METADATA",
        "lib/python3.12/site-packages/matplotlib-3.9.4.dist-info/METADATA",
        "lib/python3.12/site-packages/seaborn-0.13.2.dist-info/METADATA",
    }.issubset(material)
    assert all(not Path(relative).is_absolute() for relative in material)
    assert all(path.is_file() and not path.is_symlink() for path in material.values())


def test_discovery_requires_one_eddy_qc_distribution_metadata_file(
    tmp_path: Path,
) -> None:
    missing = _fake_fsl(
        tmp_path / "missing",
        include_openmp=False,
        omit="lib/python3.12/site-packages/eddy_qc-1.4.2.dist-info/METADATA",
    )
    duplicate = _fake_fsl(tmp_path / "duplicate", include_openmp=False)
    second = (
        duplicate
        / "lib"
        / "python3.12"
        / "site-packages"
        / "eddy_qc-9.9.9.dist-info"
        / "METADATA"
    )
    second.parent.mkdir()
    second.write_text(
        "Metadata-Version: 2.1\nName: eddy_qc\nVersion: 9.9.9\n",
        encoding="utf-8",
    )

    with pytest.raises(FSLDiscoveryError, match="distribution metadata"):
        discover_fsl(SimpleNamespace(fsldir=missing))
    with pytest.raises(FSLDiscoveryError, match="distribution metadata"):
        discover_fsl(SimpleNamespace(fsldir=duplicate))


@pytest.mark.parametrize(
    "distribution",
    tuple(name for name, _version in _EDDY_QUAD_DEPENDENCIES),
)
def test_discovery_requires_each_eddy_quad_dependency_package(
    tmp_path: Path,
    distribution: str,
) -> None:
    fsldir = _fake_fsl(
        tmp_path / distribution,
        include_openmp=False,
        omit=f"{_PYTHON_SITE_PACKAGES}/{distribution}",
    )

    with pytest.raises(FSLDiscoveryError, match=distribution):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


@pytest.mark.parametrize(("distribution", "version"), _EDDY_QUAD_DEPENDENCIES)
@pytest.mark.parametrize("installed_metadata", ("missing", "duplicate"))
def test_discovery_requires_unique_eddy_quad_dependency_distribution_metadata(
    tmp_path: Path,
    distribution: str,
    version: str,
    installed_metadata: str,
) -> None:
    metadata = (
        f"{_PYTHON_SITE_PACKAGES}/"
        f"{distribution}-{version}.dist-info/METADATA"
    )
    fsldir = _fake_fsl(
        tmp_path / f"{distribution}-{installed_metadata}",
        include_openmp=False,
        omit=metadata if installed_metadata == "missing" else None,
    )
    if installed_metadata == "duplicate":
        second = (
            fsldir
            / _PYTHON_SITE_PACKAGES
            / f"{distribution}-99.0.dist-info"
            / "METADATA"
        )
        second.parent.mkdir()
        second.write_text(
            (
                "Metadata-Version: 2.1\n"
                f"Name: {distribution}\n"
                "Version: 99.0\n"
            ),
            encoding="utf-8",
        )

    with pytest.raises(
        FSLDiscoveryError,
        match=(
            rf"{distribution}.*distribution metadata|"
            rf"distribution metadata.*{distribution}"
        ),
    ):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_symlink_in_eddy_quad_dependency_tree(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    module = (
        fsldir
        / _PYTHON_SITE_PACKAGES
        / "matplotlib"
        / "mpl-data"
        / "matplotlibrc"
    )
    external = tmp_path / "external-matplotlibrc"
    external.write_text("backend: agg\n", encoding="utf-8")
    module.unlink()
    module.symlink_to(external)

    with pytest.raises(FSLDiscoveryError, match="symbolic|regular"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_selected_eddy_replaced_after_prefix_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    eddy = fsldir / "bin" / "eddy"
    eddy.write_bytes(b"\x7fELFsynthetic-native-eddy\n")
    eddy.chmod(0o755)
    replacement = fsldir / "bin" / "replacement-eddy"
    _write_executable(
        replacement,
        "#!/usr/bin/env fslpython\nfrom fsl.base import find_cuda_exe\n",
    )
    original_open = os.open
    replacement_happened = False

    def replace_eddy_after_open(path, flags, *args, **kwargs):
        nonlocal replacement_happened
        descriptor = original_open(path, flags, *args, **kwargs)
        if not replacement_happened and Path(os.fsdecode(path)) == eddy:
            replacement.replace(eddy)
            replacement_happened = True
        return descriptor

    monkeypatch.setattr(os, "open", replace_eddy_after_open)

    with pytest.raises(
        FSLDiscoveryError,
        match="changed|during inspection",
    ):
        discover_fsl(SimpleNamespace(fsldir=fsldir))
    assert replacement_happened


def test_discovery_rejects_script_eddy_with_an_untracked_interpreter(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    _write_executable(
        fsldir / "bin" / "eddy",
        "#!/usr/bin/env python3\nprint('untracked')\n",
    )

    with pytest.raises(FSLDiscoveryError, match="eddy|fslpython|interpreter"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_fslpython_comment_spoof(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    _write_executable(
        fsldir / "bin" / "fslpython",
        (
            "#!/bin/sh\n"
            "# $FSLDIR/bin/python is intentionally only a comment\n"
            'exec /usr/bin/python3 "$@"\n'
        ),
    )

    with pytest.raises(FSLDiscoveryError, match="fslpython|bin/python"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_fslpython_unreachable_fsl_exec_spoof(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    _write_executable(
        fsldir / "bin" / "fslpython",
        (
            "#!/bin/sh\n"
            'exec /usr/bin/python3 "$@"\n'
            'exec "${FSLDIR}/bin/python" "$@"\n'
        ),
    )

    with pytest.raises(FSLDiscoveryError, match="fslpython|bin/python"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_fslpython_quoted_fsl_exec_spoof(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    _write_executable(
        fsldir / "bin" / "fslpython",
        (
            "#!/bin/sh\n"
            ": '\n"
            'exec "${FSLDIR}/bin/python" "$@"\n'
            "'\n"
            'exec /usr/bin/python3 "$@"\n'
        ),
    )

    with pytest.raises(FSLDiscoveryError, match="fslpython|bin/python"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_eddy_quad_with_an_untracked_interpreter(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    _write_executable(
        fsldir / "bin" / "eddy_quad",
        '#!/bin/sh\nexec /usr/bin/python3 "$@"\n',
    )

    with pytest.raises(FSLDiscoveryError, match="eddy_quad|bin/python"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_fslpython_replaced_after_prefix_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    launcher = fsldir / "bin" / "fslpython"
    replacement = fsldir / "bin" / "replacement-fslpython"
    _write_executable(
        replacement,
        '#!/bin/sh\nexec "${FSLDIR}/bin/python" "$@"\n',
    )
    original_open = os.open
    replacement_happened = False

    def replace_fslpython_after_open(path, flags, *args, **kwargs):
        nonlocal replacement_happened
        descriptor = original_open(path, flags, *args, **kwargs)
        if not replacement_happened and Path(os.fsdecode(path)) == launcher:
            replacement.replace(launcher)
            replacement_happened = True
        return descriptor

    monkeypatch.setattr(os, "open", replace_fslpython_after_open)

    with pytest.raises(FSLDiscoveryError, match="changed|during inspection"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))
    assert replacement_happened


@pytest.mark.parametrize(
    "distribution",
    tuple(name for name, _version in _EDDY_QUAD_DEPENDENCIES),
)
def test_discovery_rejects_more_than_1024_material_files_per_dependency(
    tmp_path: Path,
    distribution: str,
) -> None:
    fsldir = _fake_fsl(tmp_path / distribution, include_openmp=False)
    package = fsldir / _PYTHON_SITE_PACKAGES / distribution
    for index in range(1025):
        (package / f"generated_{index:04d}.py").write_text(
            f"{index}\n",
            encoding="utf-8",
        )

    with pytest.raises(
        FSLDiscoveryError,
        match="bounded (?:file|entry) limit|too many|1024",
    ):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_dependency_material_file_larger_than_64_mib(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    oversized = (
        fsldir
        / _PYTHON_SITE_PACKAGES
        / "numpy"
        / "oversized-material.bin"
    )
    with oversized.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)

    with pytest.raises(
        FSLDiscoveryError,
        match="byte|size|large|64",
    ):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


@pytest.mark.parametrize(
    "relative",
    (
        "lib/python3.12/site-packages/eddy_qc/QUAD/quad.py",
        "lib/python3.12/site-packages/fsl/data/image.py",
        "lib/python3.12/site-packages/eddy_qc-1.4.2.dist-info/METADATA",
    ),
)
def test_discovery_fails_closed_when_runtime_material_is_unreadable(
    tmp_path: Path,
    relative: str,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    material = fsldir / relative
    material.chmod(0)
    try:
        with pytest.raises(FSLDiscoveryError, match="read|inspect"):
            discover_fsl(SimpleNamespace(fsldir=fsldir))
    finally:
        material.chmod(0o644)


def test_discovery_ignores_python_version_alias_of_same_eddy_quad_package(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    (fsldir / "lib" / "python3.1").symlink_to(
        "python3.12",
        target_is_directory=True,
    )

    installation = discover_fsl(SimpleNamespace(fsldir=fsldir))

    material = dict(installation.runtime_material_files)
    assert any(
        relative.startswith(
            "lib/python3.12/site-packages/eddy_qc/"
        )
        for relative in material
    )
    assert not any(
        relative.startswith("lib/python3.1/")
        for relative in material
    )


def test_discovery_tracks_retargeted_python_interpreter_alias(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    _write_executable(fsldir / "bin" / "python3.11")
    alias = fsldir / "bin" / "python"
    before = dict(
        discover_fsl(
            SimpleNamespace(fsldir=fsldir)
        ).runtime_material_files
    )
    alias.unlink()
    alias.symlink_to("python3.11")

    after = dict(
        discover_fsl(
            SimpleNamespace(fsldir=fsldir)
        ).runtime_material_files
    )

    assert "bin/python3.11" not in before
    assert "bin/python3.11" in after
    assert set(before) != set(after)


def test_discovery_does_not_read_entire_eddy_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    eddy = fsldir / "bin" / "eddy"
    original = Path.read_bytes

    def reject_unbounded_read(path: Path) -> bytes:
        if path == eddy:
            pytest.fail("EDDY launcher was read without a byte bound")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    discover_fsl(SimpleNamespace(fsldir=fsldir))


@pytest.mark.parametrize(
    "missing",
    ("bet2", "eddy_cpu", "fslsplit", "slicer", "imrm", "imglob"),
)
def test_discovery_fails_closed_when_runtime_backend_is_missing(
    tmp_path: Path, missing: str
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl", include_openmp=False)
    (fsldir / "bin" / missing).unlink()

    with pytest.raises(FSLDiscoveryError, match=missing):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_rejects_symlink_in_eddy_quad_runtime_package(
    tmp_path: Path,
) -> None:
    fsldir = _fake_fsl(tmp_path / "fsl")
    module = (
        fsldir
        / "lib"
        / "python3.12"
        / "site-packages"
        / "eddy_qc"
        / "QUAD"
        / "quad.py"
    )
    external = tmp_path / "external.py"
    external.write_text("replacement\n", encoding="utf-8")
    module.unlink()
    module.symlink_to(external)

    with pytest.raises(FSLDiscoveryError, match="symbolic|regular"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


@pytest.mark.parametrize(
    "missing",
    [
        "topup",
        "applytopup",
        "bet",
        "fslmaths",
        "eddy",
        "eddy_quad",
        "flirt",
        "fnirt",
        "invwarp",
        "applywarp",
        "etc/flirtsch/b02b0.cnf",
        "etc/flirtsch/b02b0_1.cnf",
        "etc/flirtsch/FA_2_FMRIB58_1mm.cnf",
        "data/standard/FMRIB58_FA_1mm.nii.gz",
    ],
)
def test_discovery_rejects_missing_tools_and_resources(tmp_path: Path, missing: str):
    include_openmp = missing != "eddy"
    fsldir = _fake_fsl(
        tmp_path / "incomplete",
        include_openmp=include_openmp,
        omit=missing,
    )
    if missing == "eddy":
        (fsldir / "bin" / "eddy_openmp").unlink(missing_ok=True)

    with pytest.raises(FSLDiscoveryError, match=missing.split("/")[-1]):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


@pytest.mark.parametrize(
    "tool",
    [
        "topup",
        "applytopup",
        "bet",
        "fslmaths",
        "eddy_quad",
        "flirt",
        "fnirt",
        "invwarp",
        "applywarp",
        "eddy alternatives",
    ],
)
def test_discovery_rejects_non_executable_required_tools(
    tmp_path: Path, tool: str
):
    fsldir = _fake_fsl(tmp_path / "non executable")
    targets = (
        (fsldir / "bin" / "eddy", fsldir / "bin" / "eddy_openmp")
        if tool == "eddy alternatives"
        else (fsldir / "bin" / tool,)
    )
    for target in targets:
        target.chmod(0o644)

    with pytest.raises(FSLDiscoveryError, match="executable regular file"):
        discover_fsl(SimpleNamespace(fsldir=fsldir))


def test_discovery_returns_environment_copy_without_mutating_process(
    fake_fsldir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PATH", "/incoming/bin")
    monkeypatch.setenv("UNRELATED_VALUE", "keep-me")
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    before = dict(os.environ)

    installation = discover_fsl(SimpleNamespace(fsldir=fake_fsldir))

    assert dict(os.environ) == before
    assert installation.environment["PATH"] == f"{fake_fsldir / 'bin'}:/incoming/bin"
    assert installation.environment["FSLDIR"] == str(fake_fsldir)
    assert installation.environment["FSLOUTPUTTYPE"] == "NIFTI_GZ"
    assert installation.environment["UNRELATED_VALUE"] == "keep-me"
    copied = installation.environment
    copied["UNRELATED_VALUE"] = "changed"
    assert installation.environment["UNRELATED_VALUE"] == "keep-me"


def test_eddy_uses_corrected_pa_and_rotated_output_inputs(fsl_context: FSLContext):
    command = build_eddy_command(fsl_context)
    assert "--imain=" + str(fsl_context.gibbs_pa) in command
    assert "--topup=" + str(fsl_context.topup_prefix) in command
    assert "--repol" in command
    assert "--data_is_shelled" in command


def test_topup_selects_no_subsampling_config_for_odd_dimension(
    fsl_context: FSLContext,
):
    context = replace(fsl_context, merged_b0_shape=(120, 120, 81, 18))
    command = build_topup_command(context)
    assert "--config=" + str(
        fsl_context.fsldir / "etc/flirtsch/b02b0_1.cnf"
    ) in command


@pytest.mark.parametrize(
    "shape",
    [
        (119, 120, 80, 18),
        (120, 119, 80, 18),
        (120, 120, 79, 18),
        (119, 119, 79, 18),
    ],
)
def test_topup_uses_no_subsampling_when_any_spatial_dimension_is_odd(
    fsl_context: FSLContext, shape: tuple[int, int, int, int]
):
    command = build_topup_command(replace(fsl_context, merged_b0_shape=shape))
    assert f"--config={fsl_context.fsldir / 'etc/flirtsch/b02b0_1.cnf'}" in command


def test_topup_uses_standard_config_when_all_spatial_dimensions_are_even(
    fsl_context: FSLContext,
):
    command = build_topup_command(fsl_context)
    assert f"--config={fsl_context.fsldir / 'etc/flirtsch/b02b0.cnf'}" in command


@pytest.mark.parametrize(
    "shape",
    [(120, 120, 80), (120, 120, 80, 0), (120, -1, 80, 18)],
)
def test_context_rejects_non_4d_or_empty_merged_b0_shape(
    fsl_context: FSLContext, shape: tuple[int, ...]
):
    with pytest.raises(ValueError, match="merged_b0_shape"):
        replace(fsl_context, merged_b0_shape=shape)


def test_topup_has_exact_pa_first_input_and_output_prefix_wiring(
    fsl_context: FSLContext,
):
    assert build_topup_command(fsl_context) == [
        str(fsl_context.installation.topup),
        f"--imain={fsl_context.merged_b0}",
        f"--datain={fsl_context.acqparams_topup}",
        f"--config={fsl_context.fsldir / 'etc/flirtsch/b02b0.cnf'}",
        f"--out={fsl_context.topup_prefix}",
        f"--iout={fsl_context.topup_corrected_b0s}",
        f"--fout={fsl_context.field_hz_prefix}",
    ]


def test_applytopup_uses_one_based_rows_jacobian_and_configured_paths(
    fsl_context: FSLContext,
):
    assert build_applytopup_command(fsl_context) == [
        str(fsl_context.installation.applytopup),
        "--imain=" + ",".join(map(str, fsl_context.applytopup_inputs)),
        f"--datain={fsl_context.acqparams_topup}",
        "--inindex=1,2",
        f"--topup={fsl_context.topup_prefix}",
        "--method=jac",
        f"--out={fsl_context.applytopup_output}",
    ]


@pytest.mark.parametrize(
    ("inputs", "indices"),
    [
        ((), ()),
        ((Path("pa.nii.gz"),), (0,)),
        ((Path("pa.nii.gz"),), (-1,)),
        ((Path("pa.nii.gz"), Path("ap.nii.gz")), (1,)),
        ((Path("pa.nii.gz"),), (True,)),
    ],
)
def test_context_rejects_invalid_applytopup_arguments(
    fsl_context: FSLContext,
    inputs: tuple[Path, ...],
    indices: tuple[int, ...],
):
    with pytest.raises(ValueError, match="applytopup"):
        replace(
            fsl_context,
            applytopup_inputs=inputs,
            applytopup_indices=indices,
        )


def test_mean_b0_and_bet_commands_have_exact_path_wiring(fsl_context: FSLContext):
    assert build_topup_mean_command(fsl_context) == [
        str(fsl_context.installation.fslmaths),
        str(fsl_context.topup_corrected_b0s),
        "-Tmean",
        str(fsl_context.hifi_nodif),
    ]
    assert build_bet_command(fsl_context) == [
        str(fsl_context.installation.bet),
        str(fsl_context.hifi_nodif),
        str(fsl_context.brain_prefix),
        "-R",
        "-f",
        "0.25",
        "-g",
        "0",
        "-m",
    ]


def test_eddy_has_exact_acquisition_gradient_mask_and_thread_wiring(
    fsl_context: FSLContext,
):
    assert build_eddy_command(fsl_context) == [
        str(fsl_context.installation.eddy),
        f"--imain={fsl_context.gibbs_pa}",
        f"--mask={fsl_context.cleaned_mask}",
        f"--acqp={fsl_context.acqparams_eddy}",
        f"--index={fsl_context.index_eddy}",
        f"--bvecs={fsl_context.bvecs}",
        f"--bvals={fsl_context.bvals}",
        f"--topup={fsl_context.topup_prefix}",
        "--repol",
        "--cnr_maps",
        "--residuals",
        "--data_is_shelled",
        f"--nthr={fsl_context.eddy_threads}",
        f"--out={fsl_context.eddy_prefix}",
    ]


@pytest.mark.parametrize("threads", [0, -1, 65, True])
def test_context_rejects_unbounded_eddy_thread_counts(
    fsl_context: FSLContext, threads: int
):
    with pytest.raises(ValueError, match="eddy_threads"):
        replace(fsl_context, eddy_threads=threads)


def test_eddy_quad_uses_eddy_acquisition_and_rotated_bvecs(
    fsl_context: FSLContext,
):
    assert fsl_context.rotated_bvecs == Path(
        str(fsl_context.eddy_prefix) + ".eddy_rotated_bvecs"
    )
    assert build_eddy_quad_command(fsl_context) == [
        str(fsl_context.installation.eddy_quad),
        str(fsl_context.eddy_prefix),
        "-idx",
        str(fsl_context.index_eddy),
        "-par",
        str(fsl_context.acqparams_eddy),
        "-m",
        str(fsl_context.cleaned_mask),
        "-b",
        str(fsl_context.bvals),
        "-g",
        str(fsl_context.rotated_bvecs),
        "-f",
        str(fsl_context.field_hz_image),
        "-o",
        str(fsl_context.eddy_quad_output),
    ]
    assert str(fsl_context.bvecs) not in build_eddy_quad_command(fsl_context)


def test_jhu_commands_preserve_registration_order_and_label_interpolation(
    fsl_context: FSLContext,
):
    standard = fsl_context.fsldir / "data/standard/FMRIB58_FA_1mm.nii.gz"
    config = fsl_context.fsldir / "etc/flirtsch/FA_2_FMRIB58_1mm.cnf"
    assert build_jhu_commands(fsl_context) == [
        [
            str(fsl_context.installation.flirt),
            "-in",
            str(fsl_context.subject_fa),
            "-ref",
            str(standard),
            "-out",
            str(fsl_context.affine_fa),
            "-omat",
            str(fsl_context.affine_matrix),
            "-dof",
            "12",
            "-cost",
            "corratio",
        ],
        [
            str(fsl_context.installation.fnirt),
            f"--in={fsl_context.subject_fa}",
            f"--ref={standard}",
            f"--aff={fsl_context.affine_matrix}",
            f"--cout={fsl_context.forward_warp}",
            f"--iout={fsl_context.nonlinear_fa}",
            f"--config={config}",
        ],
        [
            str(fsl_context.installation.invwarp),
            f"--warp={fsl_context.forward_warp}",
            f"--ref={fsl_context.subject_fa}",
            f"--out={fsl_context.inverse_warp}",
        ],
        [
            str(fsl_context.installation.applywarp),
            f"--in={fsl_context.atlas_labels}",
            f"--ref={fsl_context.subject_fa}",
            f"--warp={fsl_context.inverse_warp}",
            f"--out={fsl_context.subject_atlas}",
            "--interp=nn",
        ],
    ]


def test_command_builders_return_fresh_defensive_lists(fsl_context: FSLContext):
    first = build_eddy_command(fsl_context)
    first[0] = "mutated"
    jhu_first = build_jhu_commands(fsl_context)
    jhu_first[0][0] = "mutated"

    assert build_eddy_command(fsl_context)[0] == str(fsl_context.installation.eddy)
    assert build_jhu_commands(fsl_context)[0][0] == str(
        fsl_context.installation.flirt
    )


def test_context_rejects_output_paths_that_overwrite_inputs(fsl_context: FSLContext):
    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, eddy_prefix=fsl_context.bvecs)
    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, subject_atlas=fsl_context.atlas_labels)


def test_context_rejects_same_bet_input_and_output_prefix(fsl_context: FSLContext):
    with pytest.raises(ValueError, match="output path collision"):
        replace(fsl_context, brain_prefix=fsl_context.hifi_nodif)


def test_context_rejects_nifti_input_and_extensionless_output_alias(
    fsl_context: FSLContext,
):
    raw_prefix = Path(str(fsl_context.gibbs_pa).removesuffix(".nii.gz"))

    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, eddy_prefix=raw_prefix)


def test_context_rejects_raw_dotdot_components_before_normalization(
    fsl_context: FSLContext,
):
    unsafe = (
        fsl_context.eddy_quad_output.parent
        / "nested"
        / ".."
        / fsl_context.eddy_quad_output.name
    )

    with pytest.raises(ValueError, match=r"\.\."):
        replace(fsl_context, eddy_quad_output=unsafe)


def test_context_rejects_symlink_alias_to_upstream_input(
    fsl_context: FSLContext, tmp_path: Path
):
    raw = fsl_context.gibbs_pa
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"synthetic upstream input")
    alias = tmp_path / "alias-to-pa.nii.gz"
    alias.symlink_to(raw)

    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, subject_atlas=alias)


def test_context_rejects_generated_bet_mask_as_cleaned_mask(
    fsl_context: FSLContext,
):
    raw_bet_mask = Path(f"{fsl_context.brain_prefix}_mask.nii.gz")

    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, cleaned_mask=raw_bet_mask)


@pytest.mark.parametrize(
    ("field", "suffix"),
    [
        ("topup_prefix", ".nii"),
        ("topup_prefix", ".nii.gz"),
        ("brain_prefix", ".nii"),
        ("brain_prefix", ".nii.gz"),
        ("eddy_prefix", ".nii"),
        ("eddy_prefix", ".nii.gz"),
    ],
)
def test_context_requires_sidecar_generating_prefixes_to_be_extensionless(
    fsl_context: FSLContext, field: str, suffix: str
):
    with pytest.raises(ValueError, match="extensionless"):
        replace(fsl_context, **{field: Path(f"{getattr(fsl_context, field)}{suffix}")})


def test_context_rejects_suffix_bearing_topup_prefix_with_literal_movpar_alias(
    fsl_context: FSLContext,
):
    prefix = Path(f"{fsl_context.topup_prefix}.nii.gz")
    literal_movpar = Path(f"{prefix}_movpar.txt")

    with pytest.raises(ValueError, match="extensionless"):
        replace(
            fsl_context,
            topup_prefix=prefix,
            affine_matrix=literal_movpar,
        )


def test_context_rejects_suffix_bearing_bet_prefix_with_literal_mask_alias(
    fsl_context: FSLContext,
):
    prefix = Path(f"{fsl_context.brain_prefix}.nii")
    literal_mask = Path(f"{prefix}_mask.nii.gz")

    with pytest.raises(ValueError, match="extensionless"):
        replace(
            fsl_context,
            brain_prefix=prefix,
            cleaned_mask=literal_mask,
        )


def test_context_rejects_suffix_bearing_eddy_prefix_with_literal_bvec_alias(
    fsl_context: FSLContext,
):
    prefix = Path(f"{fsl_context.eddy_prefix}.nii.gz")
    literal_rotated_bvecs = Path(f"{prefix}.eddy_rotated_bvecs")

    with pytest.raises(ValueError, match="extensionless"):
        replace(
            fsl_context,
            eddy_prefix=prefix,
            subject_atlas=literal_rotated_bvecs,
        )


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("applytopup_output", "topup_corrected_b0s"),
        ("field_hz_prefix", "hifi_nodif"),
        ("subject_atlas", "rotated_bvecs"),
    ],
)
def test_context_rejects_materialized_output_output_aliases(
    fsl_context: FSLContext, field: str, alias: str
):
    source = getattr(fsl_context, alias)
    if field != "subject_atlas":
        source = Path(f"{source}.nii.gz")

    with pytest.raises(ValueError, match="output path collision"):
        replace(fsl_context, **{field: source})


@pytest.mark.parametrize(
    "installation_path",
    [
        "topup",
        "applytopup",
        "bet",
        "fslmaths",
        "eddy",
        "eddy_quad",
        "flirt",
        "fnirt",
        "invwarp",
        "applywarp",
        "b02b0_config",
        "b02b0_no_subsampling_config",
        "fa_to_standard_config",
        "standard_fa",
    ],
)
def test_context_protects_every_fsl_installation_executable_and_resource(
    fsl_context: FSLContext, installation_path: str
):
    protected = getattr(fsl_context.installation, installation_path)
    update = (
        {
            "subject_atlas": Path(
                str(protected).removesuffix(".nii.gz")
            )
        }
        if installation_path == "standard_fa"
        else {"affine_matrix": protected}
    )

    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, **update)


def test_context_rejects_output_nifti_variant_hardlinked_to_upstream_input(
    fsl_context: FSLContext,
):
    fsl_context.gibbs_pa.parent.mkdir(parents=True)
    fsl_context.gibbs_pa.write_bytes(b"synthetic upstream NIfTI")
    materialized_output = Path(f"{fsl_context.eddy_prefix}.nii.gz")
    materialized_output.parent.mkdir(parents=True)
    os.link(fsl_context.gibbs_pa, materialized_output)

    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context)


def test_context_rejects_hardlinked_output_output_materializations(
    fsl_context: FSLContext,
):
    hifi_image = Path(f"{fsl_context.hifi_nodif}.nii.gz")
    hifi_image.parent.mkdir(parents=True)
    hifi_image.write_bytes(b"synthetic output")
    fsl_context.subject_atlas.parent.mkdir(parents=True)
    os.link(hifi_image, fsl_context.subject_atlas)

    with pytest.raises(ValueError, match="output path collision"):
        replace(fsl_context)


def test_context_rejects_output_hardlinked_to_installation_config(
    fsl_context: FSLContext, tmp_path: Path
):
    linked_matrix = tmp_path / "linked-config.mat"
    os.link(fsl_context.installation.fa_to_standard_config, linked_matrix)

    with pytest.raises(ValueError, match="raw or upstream input"):
        replace(fsl_context, affine_matrix=linked_matrix)


def test_context_accepts_distinct_sequential_products(fsl_context: FSLContext):
    assert build_topup_mean_command(fsl_context)[1:] == [
        str(fsl_context.topup_corrected_b0s),
        "-Tmean",
        str(fsl_context.hifi_nodif),
    ]
    assert build_bet_command(fsl_context)[1:3] == [
        str(fsl_context.hifi_nodif),
        str(fsl_context.brain_prefix),
    ]


def _python_fake(path: Path, source: str) -> Path:
    interpreter = Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "python"
    if not interpreter.is_file():
        interpreter = Path(os.sys.executable)
    _write_executable(path, f"#!{interpreter}\n{source}")
    return path


def test_run_fsl_command_logs_stdout_stderr_spaces_and_literal_arguments(
    tmp_path: Path,
):
    executable = _python_fake(
        tmp_path / "fake tools" / "fake fsl",
        "import json, os, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "print('ENV=' + os.environ['SUBJECT_LOCAL'], file=sys.stderr)\n",
    )
    marker = tmp_path / "must not exist"
    argv = [
        str(executable),
        "path with spaces",
        "$HOME",
        "*",
        f"$(touch {marker})",
    ]
    log_path = tmp_path / "subject output" / "logs" / "stage.log"
    process_before = dict(os.environ)

    completed = run_fsl_command(
        argv,
        log_path,
        {"PATH": os.environ.get("PATH", ""), "SUBJECT_LOCAL": "yes"},
    )

    assert completed.returncode == 0
    assert dict(os.environ) == process_before
    assert not marker.exists()
    log = log_path.read_text(encoding="utf-8")
    assert "=== FSL COMMAND START ===" in log
    assert "ARGV_JSON=" + json.dumps(argv, ensure_ascii=False) in log
    assert json.dumps(argv[1:], ensure_ascii=False) in log
    assert "ENV=yes" in log
    assert "EXIT_CODE=0" in log
    assert "=== FSL COMMAND END ===" in log


def test_run_fsl_command_appends_boundaries_instead_of_overwriting(tmp_path: Path):
    executable = _python_fake(tmp_path / "fake", "print('new output')\n")
    log_path = tmp_path / "logs" / "stage.log"
    log_path.parent.mkdir()
    log_path.write_text("existing output\n", encoding="utf-8")

    run_fsl_command([str(executable)], log_path, {})

    log = log_path.read_text(encoding="utf-8")
    assert log.startswith("existing output\n")
    assert log.count("=== FSL COMMAND START ===") == 1
    assert "new output\n" in log


def test_run_fsl_command_nonzero_preserves_outputs_and_raises_actionable_error(
    tmp_path: Path,
):
    executable = _python_fake(
        tmp_path / "fake fail",
        "import sys\nprint('partial output')\nprint('failure detail', file=sys.stderr)\nsys.exit(7)\n",
    )
    stage_result = tmp_path / "stage" / "existing.nii.gz"
    stage_result.parent.mkdir()
    stage_result.write_bytes(b"existing stage result")
    log_path = tmp_path / "logs" / "stage.log"

    with pytest.raises(ExternalCommandError) as captured:
        run_fsl_command([str(executable)], log_path, {})

    assert captured.value.executable == executable
    assert captured.value.exit_code == 7
    assert captured.value.log_path == log_path
    assert str(executable) in str(captured.value)
    assert str(log_path) in str(captured.value)
    assert stage_result.read_bytes() == b"existing stage result"
    log = log_path.read_text(encoding="utf-8")
    assert "partial output" in log
    assert "failure detail" in log
    assert "EXIT_CODE=7" in log


def test_run_fsl_command_rejects_missing_and_non_executable_files(tmp_path: Path):
    log_path = tmp_path / "logs" / "stage.log"
    missing = tmp_path / "missing"
    with pytest.raises(ExternalCommandError, match="does not exist"):
        run_fsl_command([str(missing)], log_path, {})

    non_executable = tmp_path / "not executable"
    non_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    with pytest.raises(ExternalCommandError, match="not executable"):
        run_fsl_command([str(non_executable)], log_path, {})


@pytest.mark.parametrize("argv", [[], "", ["/bin/echo", 4], [None]])
def test_run_fsl_command_rejects_invalid_argv_types(tmp_path: Path, argv):
    with pytest.raises((TypeError, ValueError), match="argv"):
        run_fsl_command(argv, tmp_path / "stage.log", {})


def test_run_fsl_command_refuses_symlink_log_without_touching_target(
    tmp_path: Path,
):
    executable = _python_fake(tmp_path / "fake", "print('unsafe write')\n")
    target = tmp_path / "outside.log"
    target.write_text("protected\n", encoding="utf-8")
    log_path = tmp_path / "logs" / "stage.log"
    log_path.parent.mkdir()
    log_path.symlink_to(target)

    with pytest.raises(ExternalCommandError, match="symlink"):
        run_fsl_command([str(executable)], log_path, {})

    assert target.read_text(encoding="utf-8") == "protected\n"


def test_run_fsl_command_refuses_symlink_log_parent_without_touching_outside(
    tmp_path: Path,
):
    executable = _python_fake(tmp_path / "fake", "print('unsafe write')\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "subject output"
    linked_parent.symlink_to(outside, target_is_directory=True)
    log_path = linked_parent / "logs" / "stage.log"

    with pytest.raises(ExternalCommandError, match="symlink"):
        run_fsl_command([str(executable)], log_path, {})

    assert list(outside.iterdir()) == []


def test_run_fsl_command_rejects_dotdot_log_path_without_creating_log(
    tmp_path: Path,
):
    executable = _python_fake(tmp_path / "fake", "print('unsafe write')\n")
    log_path = tmp_path / "logs" / ".." / "outside.log"

    with pytest.raises(ExternalCommandError, match=r"\.\."):
        run_fsl_command([str(executable)], log_path, {})

    assert not (tmp_path / "outside.log").exists()


def test_run_fsl_command_normalizes_invalid_executable_format(tmp_path: Path):
    executable = tmp_path / "invalid format"
    executable.write_text("this is not an executable format\n", encoding="utf-8")
    executable.chmod(0o755)
    log_path = tmp_path / "logs" / "stage.log"

    with pytest.raises(ExternalCommandError, match="could not launch") as captured:
        run_fsl_command([str(executable)], log_path, {})

    assert captured.value.exit_code is None
    log = log_path.read_text(encoding="utf-8")
    assert "LAUNCH_ERROR=OSError" in log
    assert "EXIT_CODE=not-launched" in log


def test_run_fsl_command_normalizes_invalid_environment_mapping(tmp_path: Path):
    executable = _python_fake(tmp_path / "fake", "print('must not run')\n")
    log_path = tmp_path / "logs" / "stage.log"

    with pytest.raises(ExternalCommandError, match="environment"):
        run_fsl_command([str(executable)], log_path, {"INVALID": None})


def test_run_fsl_command_refuses_hardlinked_log_without_touching_target(
    tmp_path: Path,
):
    executable = _python_fake(tmp_path / "fake", "print('unsafe write')\n")
    target = tmp_path / "outside.log"
    target.write_text("protected\n", encoding="utf-8")
    log_path = tmp_path / "logs" / "stage.log"
    log_path.parent.mkdir()
    os.link(target, log_path)

    with pytest.raises(ExternalCommandError, match=r"hard.?link|multiple links"):
        run_fsl_command([str(executable)], log_path, {})

    assert target.read_text(encoding="utf-8") == "protected\n"


def test_public_fsl_interfaces_are_exported():
    import dmri_pipeline

    assert dmri_pipeline.FSLContext is FSLContext
    assert dmri_pipeline.discover_fsl is discover_fsl
    assert dmri_pipeline.run_fsl_command is run_fsl_command
