from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
ATLAS_RELATIVE = "resources/jhu_48roi/JHU-ICBM-labels-2mm.nii.gz"
ATLAS_SHA256 = "974a0fd72d1214a29e58ccf33cf5aec989d937d999ae65f389dd6b3e1ffdbbad"
OLD_UNLICENSED_ERFI_SHA256 = (
    "0be34b64c408cdbe14c5f6299c22dc25c8c99df15391dc635f67e33c594ffdf6"
)
REQUIRED_DOCUMENTS = (
    "README.md",
    "docs/INSTALL_ROCKY.md",
    "docs/INPUTS.md",
    "docs/PIPELINE.md",
    "docs/OUTPUTS.md",
    "docs/QC_AND_EXCLUSION.md",
    "docs/TROUBLESHOOTING.md",
    "docs/REMOTE_VSCODE.md",
)
STAGE_ORDER = (
    "00_input_audit",
    "00_pre_denoise_motion_qc",
    "01_denoise",
    "02_gibbs",
    "03_topup",
    "04_bet",
    "05_eddy",
    "06_dti",
    "07_dki",
    "07_dki_direct",
    "08_noddi",
    "09_jhu_48roi",
    "10_summary",
    "qc",
    "report",
)


def _audit(root: Path):
    from dmri_pipeline.package_audit import audit_package

    return audit_package(root)


def _audit_error():
    from dmri_pipeline.package_audit import PackageAuditError

    return PackageAuditError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_package_root(tmp_path: Path) -> Path:
    root = tmp_path / "portable-package"
    atlas = root / ATLAS_RELATIVE
    atlas.parent.mkdir(parents=True)
    shutil.copyfile(PACKAGE_ROOT / ATLAS_RELATIVE, atlas)
    (root / "notes.txt").write_text(
        "Passwordless tokenization is discussed without assigning credentials.\n",
        encoding="utf-8",
    )
    return root


def _finding_categories(audit) -> set[str]:
    return {finding.category for finding in audit.forbidden_matches}


def _export_tracked_package_root(tmp_path: Path) -> Path:
    root = tmp_path / "exported-package"
    tracked_paths = subprocess.run(
        ["git", "-C", str(PACKAGE_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for encoded_path in tracked_paths:
        if not encoded_path:
            continue
        relative = Path(os.fsdecode(encoded_path))
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE_ROOT / relative, destination)
    return root


def test_exported_package_audit_is_clean_and_contains_only_the_historical_atlas(
    tmp_path: Path,
) -> None:
    exported_root = _export_tracked_package_root(tmp_path)
    audit = _audit(exported_root)

    assert audit.nifti_files == [ATLAS_RELATIVE]
    assert audit.forbidden_matches == []
    assert audit.cache_files == []
    assert audit.log_files == []
    assert audit.compiled_binaries == []
    assert audit.executables == ["run_eddy_batch.sh", "run_pipeline.sh", "setup_rocky.sh"]
    assert audit.sha256_by_path[ATLAS_RELATIVE] == ATLAS_SHA256
    assert audit.files == sorted(audit.files)


def test_audit_output_is_deterministic_and_uses_sorted_relative_paths(
    tmp_path: Path,
) -> None:
    root = _clean_package_root(tmp_path)
    (root / "z-last.txt").write_text("z\n", encoding="utf-8")
    (root / "a-first.txt").write_text("a\n", encoding="utf-8")

    first = _audit(root)
    second = _audit(root)

    assert first == second
    assert first.files == sorted(first.files)
    assert all(not Path(relative).is_absolute() for relative in first.files)
    assert first.manifest_sha256 == second.manifest_sha256


def test_audit_rejects_symlinks_without_following_them(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    escaped = tmp_path / "escaped.txt"
    escaped.write_text("outside\n", encoding="utf-8")
    (root / "link.txt").symlink_to(escaped)

    with pytest.raises(_audit_error(), match="symbolic link"):
        _audit(root)


def test_audit_rejects_symlinked_root(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    alias = tmp_path / "package-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(_audit_error(), match="root.*symbolic|symbolic.*root"):
        _audit(alias)


def test_audit_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    root = _clean_package_root(real_parent)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(_audit_error(), match="ancestor|symbolic"):
        _audit(alias_parent / root.name)


def test_audit_rejects_lexical_parent_traversal(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    lexical = root / "child" / ".."

    with pytest.raises(_audit_error(), match="lexical|parent traversal"):
        _audit(lexical)


def test_audit_rejects_fifo_and_hardlinked_files(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    fifo = root / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(_audit_error(), match="special|regular"):
        _audit(root)
    fifo.unlink()

    source = root / "source.txt"
    source.write_text("one identity\n", encoding="utf-8")
    os.link(source, root / "alias.txt")
    with pytest.raises(_audit_error(), match="hard-link"):
        _audit(root)


def test_audit_rejects_unreadable_file_by_mode(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    unreadable = root / "unreadable.txt"
    unreadable.write_text("hidden\n", encoding="utf-8")
    unreadable.chmod(0)
    try:
        with pytest.raises(_audit_error(), match="readable"):
            _audit(root)
    finally:
        unreadable.chmod(0o600)


def test_audit_rejects_hard_link_added_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clean_package_root(tmp_path)
    victim = root / "victim.txt"
    victim.write_text("original\n", encoding="utf-8")
    real_read = os.read
    linked = False

    def linking_read(descriptor, count):
        nonlocal linked
        content = real_read(descriptor, count)
        if not linked and content == b"original\n":
            os.link(victim, root / "late-link.txt")
            linked = True
        return content

    monkeypatch.setattr(os, "read", linking_read)
    with pytest.raises(_audit_error(), match="changed|hard-link|link"):
        _audit(root)
    assert linked


def test_audit_rejects_directory_entries_added_during_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clean_package_root(tmp_path)
    watched = root / "watched"
    watched.mkdir()
    (watched / "first.txt").write_text("first\n", encoding="utf-8")
    real_scandir = os.scandir
    changed = False

    def changing_scandir(path):
        nonlocal changed
        entries = list(real_scandir(path))
        if not changed and any(entry.name == "first.txt" for entry in entries):
            (watched / "late.txt").write_text("late\n", encoding="utf-8")
            changed = True
        return iter(entries)

    monkeypatch.setattr(os, "scandir", changing_scandir)
    with pytest.raises(_audit_error(), match="directory.*changed|changed.*directory"):
        _audit(root)
    assert changed


def test_audit_rejects_file_replaced_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clean_package_root(tmp_path)
    victim = root / "victim.txt"
    victim.write_text("original\n", encoding="utf-8")
    replacement = root / "replacement.txt"
    replacement.write_text("replacement\n", encoding="utf-8")
    real_open = os.open
    replaced = False

    def replacing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if not replaced and os.fsdecode(path) == victim.name:
            replacement.replace(victim)
            replaced = True
        return descriptor

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(_audit_error(), match="changed|identity"):
        _audit(root)
    assert replaced


def test_audit_reports_binary_magic_and_forbidden_suffixes(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    (root / "renamed-data.bin").write_bytes(b"\x7fELF" + b"\0" * 32)
    (root / "results.csv").write_text("subject,value\nSYN001,1\n", encoding="utf-8")
    (root / "worker.log").write_text("runtime output\n", encoding="utf-8")
    (root / "checkpoint.mat").write_bytes(b"MATLAB 5.0 MAT-file")

    audit = _audit(root)

    assert audit.compiled_binaries == ["renamed-data.bin"]
    assert audit.log_files == ["worker.log"]
    assert {"compiled_binary", "generated_result", "runtime_log", "mat_file"}.issubset(
        _finding_categories(audit)
    )


def test_audit_reports_broad_binary_archive_and_cache_signatures(
    tmp_path: Path,
) -> None:
    root = _clean_package_root(tmp_path)
    samples = {
        "figure.dat": b"\x89PNG\r\n\x1a\n",
        "paper.dat": b"%PDF-1.7\n",
        "archive.dat": b"PK\x03\x04",
        "array.dat": b"\x89HDF\r\n\x1a\n",
        "module.dat": b"\0asm",
        "bytecode.dat": b"\xca\xfe\xba\xbe",
        "library.dat": b"!<arch>\n",
    }
    for name, content in samples.items():
        (root / name).write_bytes(content)
    (root / "bundle.zip").write_text("archive by suffix\n", encoding="utf-8")
    (root / ".DS_Store").write_text("finder cache\n", encoding="utf-8")
    (root / ".coverage").write_text("coverage cache\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir()
    (git / "config").write_text("[core]\n", encoding="utf-8")

    audit = _audit(root)
    categories = _finding_categories(audit)

    assert {"generated_result", "archive", "binary_data", "compiled_binary", "cache"}.issubset(
        categories
    )


def test_audit_decodes_percent_hex_unicode_escapes_and_nfkc(
    tmp_path: Path,
) -> None:
    root = _clean_package_root(tmp_path)
    private_path = bytes(
        (47, 85, 115, 101, 114, 115, 47, 101, 120, 97, 109, 112, 108, 101, 47)
    )
    patient_id = bytes((67, 79, 72, 79, 82, 84, 55, 51, 49))
    percent_encoded = "".join(f"%{value:02x}" for value in private_path)
    hex_escaped = "".join(f"\\x{value:02x}" for value in patient_id)
    unicode_escaped = "".join(f"\\u{value:04x}" for value in private_path)
    full_width = "".join(chr(value + 0xFEE0) for value in patient_id)
    (root / "encoded.txt").write_text(
        "\n".join((percent_encoded, hex_escaped, unicode_escaped, full_width)),
        encoding="utf-8",
    )

    audit = _audit(root)

    assert {"private_path", "patient_identifier"}.issubset(
        _finding_categories(audit)
    )


def test_audit_decodes_bounded_bare_hex_and_base64_markers(
    tmp_path: Path,
) -> None:
    root = _clean_package_root(tmp_path)
    private_path = bytes(
        (47, 85, 115, 101, 114, 115, 47, 101, 120, 97, 109, 112, 108, 101, 47)
    )
    patient_id = bytes((67, 79, 72, 79, 82, 84, 55, 51, 49))
    (root / "encoded-tokens.txt").write_text(
        patient_id.hex() + "\n" + base64.b64encode(private_path).decode("ascii"),
        encoding="utf-8",
    )

    audit = _audit(root)

    assert {"private_path", "patient_identifier"}.issubset(
        _finding_categories(audit)
    )


def test_audit_detects_split_private_markers_and_credential_assignments(
    tmp_path: Path,
) -> None:
    root = _clean_package_root(tmp_path)
    private_path = bytes(
        (47, 85, 115, 101, 114, 115, 47, 101, 120, 97, 109, 112, 108, 101, 47)
    )
    patient_id = bytes((67, 79, 72, 79, 82, 84, 55, 51, 49))
    credential = bytes(
        (97, 112, 105, 95, 116, 111, 107, 101, 110, 32, 61, 32, 34, 115, 101, 99, 114, 101, 116, 34)
    )
    (root / "split.txt").write_bytes(
        b'"'
        + private_path[:7]
        + b'" + "'
        + private_path[7:]
        + b'"\n"'
        + patient_id[:2]
        + b'" + "'
        + patient_id[2:]
        + b'"\n'
        + credential
        + b"\n"
    )

    audit = _audit(root)

    assert {"private_path", "patient_identifier", "credential"}.issubset(
        _finding_categories(audit)
    )


def test_audit_avoids_benign_token_and_password_words(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)

    audit = _audit(root)

    assert audit.forbidden_matches == []


def test_audit_scans_normalized_relative_file_and_directory_names(
    tmp_path: Path,
) -> None:
    root = _clean_package_root(tmp_path)
    patient_id = bytes((67, 79, 72, 79, 82, 84, 55, 51, 49)).decode("ascii")
    named = root / f"prefix-{patient_id}-suffix"
    named.mkdir()
    (named / "safe.txt").write_text("generic\n", encoding="utf-8")

    audit = _audit(root)

    assert "patient_identifier" in _finding_categories(audit)


def test_audit_detects_generic_subject_data_tree(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    private_tree = bytes(
        (
            115,
            117,
            98,
            106,
            101,
            99,
            116,
            115,
            47,
            115,
            116,
            117,
            100,
            121,
            100,
            97,
            116,
            97,
        )
    ).decode("ascii")
    named = root / private_tree
    named.mkdir(parents=True)
    (named / "safe.txt").write_text("generic\n", encoding="utf-8")

    audit = _audit(root)

    assert "private_path" in _finding_categories(audit)


def test_audit_is_path_independent_and_enforces_resource_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dmri_pipeline import package_audit

    first_root = _clean_package_root(tmp_path / "first")
    second_root = _clean_package_root(tmp_path / "second")
    first = _audit(first_root)
    second = _audit(second_root)
    assert first == second
    assert first.root == "."

    monkeypatch.setattr(package_audit, "_MAX_TOTAL_BYTES", first.total_bytes - 1)
    with pytest.raises(_audit_error(), match="aggregate|total"):
        _audit(first_root)


def test_audit_rejects_oversized_atlas_before_hashing(tmp_path: Path) -> None:
    root = _clean_package_root(tmp_path)
    atlas = root / ATLAS_RELATIVE
    atlas.write_bytes(atlas.read_bytes() + b"x")

    with pytest.raises(_audit_error(), match="atlas.*size|size.*atlas"):
        _audit(root)


def test_required_documentation_covers_the_public_contract() -> None:
    for relative in REQUIRED_DOCUMENTS:
        path = PACKAGE_ROOT / relative
        assert path.is_file() and path.stat().st_size > 0, relative
    assert not (PACKAGE_ROOT / "docs/INSTALL_" "MACOS.md").exists()

    readme_text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    readme = readme_text.lower()
    for required in (
        "research only",
        "not for clinical diagnosis",
        "rocky linux 9.7",
        "x86_64",
        "server",
        "dmri_software_config",
        "./setup_rocky.sh --check",
        "./run_pipeline.sh --validate-only",
        "./run_pipeline.sh config/",
        "pa",
        "ap",
        "total_readout_time",
        "hold_for_review",
        "include_with_flags",
        "exclude",
        "--force-stage",
        "runtime",
        "disk",
        "third-party",
        "reproducibility",
    ):
        assert required in readme, required

    quick_start = readme_text.split("## Quick start", 1)[1].split("##", 1)[0]
    commands = [
        line.strip()
        for line in quick_start.splitlines()
        if line.startswith(("cp ", "export ", "./"))
    ]
    assert commands == [
        "cp config/subject.example.yaml config/subject.yaml",
        "export DMRI_SOFTWARE_CONFIG=/absolute/path/to/dmri-rocky9.sh",
        "./setup_rocky.sh",
        "./run_pipeline.sh config/subject.yaml",
    ]
    for citation in (
        "10.1016/j.neuroimage.2016.08.016",
        "10.1002/mrm.26054",
        "10.1016/s1053-8119(03)00336-7",
        "10.1016/j.neuroimage.2015.10.019",
        "10.1002/mrm.20508",
        "10.1016/j.neuroimage.2012.03.072",
        "10.1016/j.neuroimage.2007.12.035",
    ):
        assert citation in readme

    pipeline = (PACKAGE_ROOT / "docs/PIPELINE.md").read_text(encoding="utf-8")
    positions = [pipeline.index(f"`{stage}`") for stage in STAGE_ORDER]
    assert positions == sorted(positions)

    troubleshooting = (
        PACKAGE_ROOT / "docs/TROUBLESHOOTING.md"
    ).read_text(encoding="utf-8").lower()
    for topic in (
        "fsldir",
        "matlab",
        "optimization toolbox",
        "mex",
        "odd",
        "eddy_quad",
        "hold_for_review",
        "exclude",
        "noddi",
        "48",
        "50",
        "--force-stage",
    ):
        assert topic in troubleshooting, topic

    outputs = (PACKAGE_ROOT / "docs/OUTPUTS.md").read_text(encoding="utf-8")
    for filename in (
        "FA.nii.gz",
        "MD.nii.gz",
        "AD.nii.gz",
        "RD.nii.gz",
        "V1.nii.gz",
        "MK.nii.gz",
        "AK.nii.gz",
        "RK.nii.gz",
        "S0.nii.gz",
        "NODDI_odi.nii",
        "NODDI_ficvf.nii",
        "NODDI_fiso.nii",
        "NODDI_kappa.nii",
        "NODDI_fmin.nii",
        "NODDI_error_code.nii",
        "NODDI_fibredirs_xvec.nii",
        "NODDI_fibredirs_yvec.nii",
        "NODDI_fibredirs_zvec.nii",
        "NODDI_params.mat",
        "17 named PNGs",
    ):
        assert filename in outputs, filename


def test_vscode_recommendations_are_valid_and_non_secret() -> None:
    payload = json.loads(
        (PACKAGE_ROOT / ".vscode" / "extensions.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "recommendations": [
            "ms-vscode-remote.remote-ssh",
            "ms-python.python",
            "ms-python.vscode-pylance",
        ]
    }
    rendered = json.dumps(payload).lower()
    for forbidden in ("hostname", "identityfile", "password", "/users/", "/home/"):
        assert forbidden not in rendered


def test_remote_vscode_document_keeps_compute_on_server() -> None:
    text = (PACKAGE_ROOT / "docs" / "REMOTE_VSCODE.md").read_text(
        encoding="utf-8"
    ).lower()
    for required in (
        "remote - ssh",
        "ssh dmri-rocky",
        "open folder",
        "rocky linux 9.7",
        "uname -m",
        "x86_64",
        "dmri_software_config",
        "source \"${dmri_software_config}\"",
        "tmux new -s",
        "tmux attach",
        "do not use sshfs",
    ):
        assert required in text, required


def test_release_metadata_is_rocky_only_and_consistent() -> None:
    assert (PACKAGE_ROOT / "VERSION").read_text(encoding="utf-8").strip() == "2.0.0"
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.0.0"' in pyproject
    for relative in (
        "README.md",
        "docs/INSTALL_ROCKY.md",
        "docs/TROUBLESHOOTING.md",
        "pyproject.toml",
    ):
        text = (PACKAGE_ROOT / relative).read_text(encoding="utf-8").lower()
        assert "macos" not in text, relative
        assert "setup_" "macos.sh" not in text, relative


def test_attribution_manifest_has_verified_internal_hashes_and_no_project_grant():
    manifest_path = PACKAGE_ROOT / "licenses/THIRD_PARTY_ATTRIBUTION.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert set(manifest["components"]) == {
        "henrique_helper",
        "noddi_v1_05",
        "nifti_matlab",
        "spm12_r7487_nifti_runtime",
        "jhu_48roi",
    }
    for component in manifest["components"].values():
        assert component["source_url"].startswith("https://")
        assert component["notice_sources"]
        assert all(
            source.startswith("https://")
            for source in component["notice_sources"].values()
        )
        assert component["files"]
        assert set(component["notices"]) == set(component["notice_sources"])
        for relative, expected in component["notices"].items():
            path = PACKAGE_ROOT / relative
            assert path.is_file(), relative
            assert _sha256(path) == expected, relative
        for relative, expected in component["files"].items():
            path = PACKAGE_ROOT / relative
            assert path.is_file(), relative
            assert _sha256(path) == expected["sha256"], relative

    erfi = (
        PACKAGE_ROOT
        / "vendor/noddi_toolbox_v1.05/models/watson/NODDI_erfi.m"
    )
    assert _sha256(erfi) != OLD_UNLICENSED_ERFI_SHA256
    noddi = manifest["components"]["noddi_v1_05"]
    assert noddi["modified"] is True
    assert "package-authored" in noddi["modifications"]["NODDI_erfi.m"].lower()
    assert noddi["commit"] is None
    assert noddi["source_archive_sha256"] is None
    retained_noddi_files = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "vendor/noddi_toolbox_v1.05").rglob("*")
        if path.is_file()
    } | {"scripts/matlab/CreateROI.m"}
    assert set(noddi["files"]) == retained_noddi_files
    assert "scripts/matlab/CreateROI.m" in noddi["modifications"]

    spm_files = set(
        manifest["components"]["spm12_r7487_nifti_runtime"]["files"]
    )
    legacy_files = set(manifest["components"]["nifti_matlab"]["files"])
    retained_nifti_files = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "vendor/nifti_matlab").rglob("*")
        if path.is_file()
    }
    assert len(spm_files) == 72
    assert len(
        {
            path
            for path in legacy_files
            if path.startswith("vendor/nifti_matlab/matlab/")
        }
    ) == 10
    assert spm_files.isdisjoint(legacy_files)
    assert spm_files | legacy_files == retained_nifti_files
    assert set(manifest["components"]["henrique_helper"]["files"]) == {
        "vendor/henrique_helpers/dki_alternative.py"
    }
    assert set(manifest["components"]["jhu_48roi"]["files"]) == {
        "resources/jhu_48roi/JHU-ICBM-labels-2mm.nii.gz",
        "resources/jhu_48roi/JHU-labels.xml",
        "resources/jhu_48roi/provenance.json",
    }
    all_component_files = [
        set(component["files"]) for component in manifest["components"].values()
    ]
    for index, component_files in enumerate(all_component_files):
        for other_files in all_component_files[index + 1 :]:
            assert component_files.isdisjoint(other_files)
    assert not (
        PACKAGE_ROOT / "vendor/nifti_matlab/matlab/make.m"
    ).exists()
    assert not (PACKAGE_ROOT / "LICENSE").exists()
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "no project-wide open-source licence" in readme


@pytest.mark.parametrize(
    ("relative", "expected_sha256"),
    (
        (
            "licenses/HENRIQUE-CC-BY-4.0.txt",
            "95df2e9564862e51d69683a899b6dcc8218d577057bdf67322880769ff85f29e",
        ),
        (
            "licenses/NODDI-LICENSE.txt",
            "395132693d689b2944785b662935ebc6a6d6f3ffc52ef0faa9e8897dfc7957ef",
        ),
        (
            "licenses/NIFTI-MATLAB-LICENSE.txt",
            "ba4eeab8eaf6c549aa5fc9b90505e5cb75f956154bead94f3aa5eefc6ffb6555",
        ),
        (
            "licenses/GPL-2.0.txt",
            "b499eddebda05a8859e32b820a64577d91f1de2b52efa2a1575a2cb4000bc259",
        ),
        (
            "licenses/ARTISTIC-2.0.txt",
            "f906736d3e8ad2237e8ff6ea0c8e41ae51ce4c5b1e0727ee25334c22b3de5b85",
        ),
        (
            "licenses/FSL-JHU-LICENSE.md",
            "216dbf45d7574ce175f5c79145cc1d8eab814b08904b93d00a0696227d4a09a0",
        ),
        (
            "licenses/SPM12-r7487-LICENCE.txt",
            "3cf9e20bb94f096bdb402da29e3d5b690dacaf4274c4bf606f5a43b26c108670",
        ),
    ),
)
def test_verbatim_notices_have_fixed_authoritative_hashes(
    relative: str,
    expected_sha256: str,
) -> None:
    assert _sha256(PACKAGE_ROOT / relative) == expected_sha256
