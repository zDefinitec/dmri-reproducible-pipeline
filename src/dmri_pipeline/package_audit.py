"""Fail-closed audit of a package tree before public distribution."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import stat
import unicodedata
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


_ATLAS_PATH = "resources/jhu_48roi/JHU-ICBM-labels-2mm.nii.gz"
_ATLAS_SHA256 = "974a0fd72d1214a29e58ccf33cf5aec989d937d999ae65f389dd6b3e1ffdbbad"
_ALLOWED_EXECUTABLES = frozenset({"run_pipeline.sh", "setup_rocky.sh"})
_MAX_DEPTH = 32
_MAX_ENTRIES = 20_000
_MAX_TEXT_BYTES = 8 * 1024 * 1024
_MAX_ATLAS_BYTES = 9_611
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_PRIVATE_HOME_PATH = re.compile(
    r"(?i)(?<![a-z0-9])(?:/users|/home)/[a-z0-9._-]+/"
)
_PRIVATE_DATA_TREE = re.compile(
    r"(?i)(?<![a-z0-9])(?:subjects?|participants?|patients?|pca)"
    r"/[a-z0-9._-]{3,}(?:/|$)"
)
_PATIENT_IDENTIFIER = re.compile(
    r"""(?x)
    (?<![A-Za-z0-9])
    (?!(?:ANON|COMPLEX|EXAMPLE|FLOAT|FSL|IEEE|MATLAB|NODDI|SAMPLE|SPM|SUB|SYN|SYNTH|TASK|TEST)[_-]?\d)
    [A-Z]{4,12}[_-]?\d{3,8}
    (?![A-Za-z0-9])
    """
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(?:api[_-]?(?:key|token)|access[_-]?token|password|passwd|client[_-]?secret)\b
    \s*(?:=|:)\s*
    (?!none\b|null\b|example\b|placeholder\b|<[^>\r\n]+>)
    ["']?[^\s"'`,;}{]{4,}
    """
)
_CACHE_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".ipynb_checkpoints",
        "__pycache__",
        "htmlcov",
    }
)
_CACHE_SUFFIXES = (".pyc", ".pyo", ".un~", ".swp", ".swo", ".tmp", "~")
_COMPILED_SUFFIXES = (
    ".class",
    ".dll",
    ".dylib",
    ".mex",
    ".mexa64",
    ".mexmaci64",
    ".mexw64",
    ".o",
    ".obj",
    ".so",
)
_GENERATED_SUFFIXES = (".csv", ".pdf", ".png")
_ARCHIVE_SUFFIXES = (".7z", ".gz", ".rar", ".tar", ".tar.gz", ".tgz", ".zip")
_MAGIC_CATEGORIES = (
    (b"\x7fELF", "compiled_binary"),
    (b"MZ", "compiled_binary"),
    (b"\xca\xfe\xba\xbe", "compiled_binary"),
    (b"\xce\xfa\xed\xfe", "compiled_binary"),
    (b"\xcf\xfa\xed\xfe", "compiled_binary"),
    (b"\xfe\xed\xfa\xce", "compiled_binary"),
    (b"\xfe\xed\xfa\xcf", "compiled_binary"),
    (b"\0asm", "compiled_binary"),
    (b"BC\xc0\xde", "compiled_binary"),
    (b"!<arch>\n", "compiled_binary"),
    (b"%PDF-", "generated_result"),
    (b"\x89PNG\r\n\x1a\n", "generated_result"),
    (b"\xff\xd8\xff", "generated_result"),
    (b"II*\0", "generated_result"),
    (b"MM\0*", "generated_result"),
    (b"PK\x03\x04", "archive"),
    (b"\x1f\x8b", "archive"),
    (b"MATLAB 5.0 MAT-file", "mat_file"),
    (b"\x89HDF\r\n\x1a\n", "binary_data"),
)


class PackageAuditError(RuntimeError):
    """Raised when the tree cannot be audited without following ambiguity."""


@dataclass(frozen=True, order=True)
class PackageFinding:
    """One forbidden distributable artifact."""

    path: str
    category: str
    detail: str


@dataclass(frozen=True)
class PackageAudit:
    """Deterministic public-package audit result."""

    root: str
    files: list[str]
    nifti_files: list[str]
    forbidden_matches: list[PackageFinding]
    cache_files: list[str]
    log_files: list[str]
    compiled_binaries: list[str]
    executables: list[str]
    sha256_by_path: dict[str, str]
    total_bytes: int
    manifest_sha256: str


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _stable_metadata(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _decode_obfuscation(text: str) -> str:
    decoded = text
    for _ in range(3):
        candidate = urllib.parse.unquote(decoded)
        candidate = re.sub(
            r"\\x([0-9a-fA-F]{2})",
            lambda match: chr(int(match.group(1), 16)),
            candidate,
        )
        candidate = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda match: chr(int(match.group(1), 16)),
            candidate,
        )
        candidate = unicodedata.normalize("NFKC", candidate)
        if candidate == decoded:
            break
        decoded = candidate
    return decoded


def _normalise_for_signatures(text: str, *, casefold: bool = True) -> bytes:
    text = _decode_obfuscation(text)
    text = re.sub(r"""["']\s*(?:\+\s*)?["']""", "", text)
    source = text.casefold() if casefold else text
    normalised = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "/._-")
        else " "
        for character in source
    )
    return " ".join(normalised.split()).encode("ascii")


def _direct_signature_categories(text: str) -> set[str]:
    decoded = _decode_obfuscation(text)
    normalised = _normalise_for_signatures(decoded).decode("ascii")
    case_preserved = _normalise_for_signatures(
        decoded, casefold=False
    ).decode("ascii")
    categories: set[str] = set()
    if _PRIVATE_HOME_PATH.search(normalised) or _PRIVATE_DATA_TREE.search(normalised):
        categories.add("private_path")
    if _PATIENT_IDENTIFIER.search(case_preserved):
        categories.add("patient_identifier")
    if _CREDENTIAL_ASSIGNMENT.search(decoded):
        categories.add("credential")
    return categories


def _signature_categories(text: str) -> set[str]:
    decoded = _decode_obfuscation(text)
    categories = _direct_signature_categories(decoded)

    def compare_decoded(candidate: bytes) -> None:
        try:
            candidate_text = candidate.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return
        categories.update(_direct_signature_categories(candidate_text))

    for match in re.finditer(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{2}){4,128}(?![0-9a-fA-F])",
        decoded,
    ):
        compare_decoded(bytes.fromhex(match.group(0)))
    for match in re.finditer(
        r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{4}){2,64}={0,2}"
        r"(?![A-Za-z0-9+/=])",
        decoded,
    ):
        try:
            candidate = base64.b64decode(match.group(0), validate=True)
        except (ValueError, binascii.Error):
            continue
        compare_decoded(candidate)
    return categories


def _magic_categories(content: bytes) -> set[str]:
    return {
        category
        for prefix, category in _MAGIC_CATEGORIES
        if content.startswith(prefix)
    }


def _classify_file(
    relative: str,
    metadata: os.stat_result,
    content: bytes,
) -> tuple[list[PackageFinding], bool, bool, bool, bool]:
    findings: list[PackageFinding] = []
    path = Path(relative)
    lower_name = path.name.casefold()
    lower_relative = relative.casefold()
    suffixes = "".join(path.suffixes).casefold()

    is_cache = (
        any(part.casefold() in _CACHE_NAMES for part in path.parts)
        or lower_name.endswith(_CACHE_SUFFIXES)
        or lower_name == ".ds_store"
        or lower_name.startswith(".coverage")
    )
    is_log = lower_name.endswith(".log")
    has_compiled_suffix = lower_name.endswith(_COMPILED_SUFFIXES)
    magic_categories = _magic_categories(content)
    is_compiled = has_compiled_suffix or "compiled_binary" in magic_categories

    if is_cache:
        findings.append(PackageFinding(relative, "cache", "cache artifact"))
    if is_log:
        findings.append(PackageFinding(relative, "runtime_log", "runtime log"))
    if is_compiled:
        findings.append(
            PackageFinding(relative, "compiled_binary", "compiled binary artifact")
        )
    if lower_name.endswith(".mat") or "mat_file" in magic_categories:
        findings.append(PackageFinding(relative, "mat_file", "MATLAB data file"))
    if (
        lower_name.endswith(_GENERATED_SUFFIXES)
        or "generated_result" in magic_categories
    ):
        findings.append(
            PackageFinding(relative, "generated_result", "generated result artifact")
        )
    if (
        suffixes.endswith(_ARCHIVE_SUFFIXES)
        and relative != _ATLAS_PATH
    ) or "archive" in magic_categories:
        findings.append(PackageFinding(relative, "archive", "archive artifact"))
    if "binary_data" in magic_categories:
        findings.append(PackageFinding(relative, "binary_data", "binary data artifact"))
    if (
        any(part.casefold() in {"outputs", "results", ".work"} for part in path.parts)
        or "checkpoint" in lower_name
    ):
        findings.append(
            PackageFinding(relative, "generated_result", "generated output path")
        )

    is_nifti = lower_relative.endswith((".nii", ".nii.gz"))
    if is_nifti and relative != _ATLAS_PATH:
        findings.append(
            PackageFinding(relative, "unexpected_nifti", "only the fixed JHU atlas is allowed")
        )

    executable = bool(metadata.st_mode & 0o111)
    if executable and relative not in _ALLOWED_EXECUTABLES:
        findings.append(
            PackageFinding(relative, "unexpected_executable", "executable bit is not allowed")
        )

    if relative != _ATLAS_PATH and not magic_categories and not is_nifti:
        if len(content) > _MAX_TEXT_BYTES:
            raise PackageAuditError(
                f"text file exceeds {_MAX_TEXT_BYTES} byte audit limit: {relative}"
            )
        try:
            text = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PackageAuditError(f"non-UTF-8 package file: {relative}") from exc
        for category in sorted(_signature_categories(text)):
            findings.append(
                PackageFinding(relative, category, "forbidden content signature")
            )

    return findings, is_nifti, is_cache, is_log, is_compiled


def _open_root_without_symlinks(root_path: Path) -> tuple[int, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(root_path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for index, component in enumerate(absolute.parts[1:]):
            try:
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise PackageAuditError(
                    f"cannot inspect package root component: {component}"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                location = "root" if index == len(absolute.parts[1:]) - 1 else "ancestor"
                raise PackageAuditError(
                    f"package {location} must not be a symbolic link"
                )
            if not stat.S_ISDIR(before.st_mode):
                raise PackageAuditError(
                    f"package root component is not a directory: {component}"
                )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise PackageAuditError(
                    f"cannot open package root component safely: {component}"
                ) from exc
            opened = os.fstat(child)
            if _stable_metadata(before) != _stable_metadata(opened):
                os.close(child)
                raise PackageAuditError(
                    f"package root component changed while opening: {component}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def audit_package(root: Path | str) -> PackageAudit:
    """Audit *root* without following symlinks or trusting path-based reads."""

    root_path = Path(root)
    if ".." in root_path.parts:
        raise PackageAuditError("package root contains lexical parent traversal")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd, root_metadata = _open_root_without_symlinks(root_path)
    except OSError as exc:
        raise PackageAuditError(f"cannot open package root safely: {root_path}") from exc

    files: list[str] = []
    nifti_files: list[str] = []
    findings: list[PackageFinding] = []
    cache_files: list[str] = []
    log_files: list[str] = []
    compiled_binaries: list[str] = []
    executables: list[str] = []
    sha256_by_path: dict[str, str] = {}
    seen_identities = {_identity(os.fstat(root_fd))}
    total_bytes = 0
    entry_count = 0

    def walk(directory_fd: int, prefix: str, depth: int) -> None:
        nonlocal entry_count, total_bytes
        if depth > _MAX_DEPTH:
            raise PackageAuditError(f"package tree exceeds maximum depth {_MAX_DEPTH}")
        directory_before = os.fstat(directory_fd)
        try:
            entries = sorted(os.scandir(directory_fd), key=lambda item: item.name)
        except OSError as exc:
            raise PackageAuditError(f"cannot enumerate package directory: {prefix or '.'}") from exc

        for entry in entries:
            entry_count += 1
            if entry_count > _MAX_ENTRIES:
                raise PackageAuditError(
                    f"package tree exceeds maximum entry count {_MAX_ENTRIES}"
                )
            name = entry.name
            relative = f"{prefix}/{name}" if prefix else name
            if name in {"", ".", ".."} or "/" in name:
                raise PackageAuditError(f"unsafe package entry name: {relative}")
            for category in sorted(_signature_categories(relative)):
                findings.append(
                    PackageFinding(
                        relative, category, "forbidden relative-path signature"
                    )
                )
            try:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise PackageAuditError(f"cannot inspect package entry: {relative}") from exc
            mode = before.st_mode
            if stat.S_ISLNK(mode):
                raise PackageAuditError(f"symbolic link is forbidden: {relative}")

            if stat.S_ISDIR(mode):
                if name.casefold() in _CACHE_NAMES:
                    cache_files.append(relative)
                    findings.append(
                        PackageFinding(relative, "cache", "cache directory")
                    )
                try:
                    child_fd = os.open(
                        name, directory_flags | nofollow, dir_fd=directory_fd
                    )
                except OSError as exc:
                    raise PackageAuditError(
                        f"cannot open package directory safely: {relative}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if _stable_metadata(before) != _stable_metadata(opened):
                        raise PackageAuditError(
                            f"directory identity changed while opening: {relative}"
                        )
                    identity = _identity(opened)
                    if identity in seen_identities:
                        raise PackageAuditError(
                            f"duplicate filesystem identity in package: {relative}"
                        )
                    seen_identities.add(identity)
                    walk(child_fd, relative, depth + 1)
                    after_open = os.fstat(child_fd)
                    after_path = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if _stable_metadata(opened) != _stable_metadata(
                        after_open
                    ) or _stable_metadata(opened) != _stable_metadata(after_path):
                        raise PackageAuditError(
                            f"directory identity changed during audit: {relative}"
                        )
                finally:
                    os.close(child_fd)
                continue

            if not stat.S_ISREG(mode):
                raise PackageAuditError(f"special non-regular file is forbidden: {relative}")
            if before.st_nlink != 1:
                raise PackageAuditError(f"hard-linked file is forbidden: {relative}")
            if before.st_mode & 0o444 == 0:
                raise PackageAuditError(f"file is not readable by mode: {relative}")
            if relative == _ATLAS_PATH and before.st_size != _MAX_ATLAS_BYTES:
                raise PackageAuditError(
                    f"atlas size must be exactly {_MAX_ATLAS_BYTES} bytes: {relative}"
                )
            identity = _identity(before)
            if identity in seen_identities:
                raise PackageAuditError(
                    f"duplicate filesystem identity in package: {relative}"
                )
            seen_identities.add(identity)

            try:
                file_fd = os.open(name, os.O_RDONLY | nofollow, dir_fd=directory_fd)
            except OSError as exc:
                raise PackageAuditError(f"cannot open package file safely: {relative}") from exc
            try:
                opened = os.fstat(file_fd)
                if _stable_metadata(before) != _stable_metadata(opened):
                    raise PackageAuditError(f"file identity changed while opening: {relative}")
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                byte_count = 0
                while True:
                    chunk = os.read(file_fd, _READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    if relative != _ATLAS_PATH:
                        chunks.append(chunk)
                    byte_count += len(chunk)
                    if total_bytes + byte_count > _MAX_TOTAL_BYTES:
                        raise PackageAuditError(
                            f"package aggregate byte limit {_MAX_TOTAL_BYTES} exceeded"
                        )
                    if relative != _ATLAS_PATH and byte_count > _MAX_TEXT_BYTES:
                        raise PackageAuditError(
                            f"text file exceeds {_MAX_TEXT_BYTES} byte audit limit: {relative}"
                        )
                after_open = os.fstat(file_fd)
                try:
                    after_path = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                except OSError as exc:
                    raise PackageAuditError(
                        f"file changed or vanished during audit: {relative}"
                    ) from exc
                if (
                    _stable_metadata(opened) != _stable_metadata(after_open)
                    or _stable_metadata(opened) != _stable_metadata(after_path)
                    or byte_count != opened.st_size
                ):
                    raise PackageAuditError(f"file changed during audit: {relative}")
            finally:
                os.close(file_fd)

            content = b"".join(chunks)
            file_digest = digest.hexdigest()
            files.append(relative)
            sha256_by_path[relative] = file_digest
            total_bytes += byte_count
            file_findings, is_nifti, is_cache, is_log, is_compiled = _classify_file(
                relative, opened, content
            )
            findings.extend(file_findings)
            if is_nifti:
                nifti_files.append(relative)
            if is_cache:
                cache_files.append(relative)
            if is_log:
                log_files.append(relative)
            if is_compiled:
                compiled_binaries.append(relative)
            if opened.st_mode & 0o111:
                executables.append(relative)

        try:
            final_names = sorted(entry.name for entry in os.scandir(directory_fd))
        except OSError as exc:
            raise PackageAuditError(
                f"cannot re-enumerate package directory: {prefix or '.'}"
            ) from exc
        directory_after = os.fstat(directory_fd)
        initial_names = [entry.name for entry in entries]
        if (
            initial_names != final_names
            or _stable_metadata(directory_before) != _stable_metadata(directory_after)
        ):
            raise PackageAuditError(
                f"package directory changed during audit: {prefix or '.'}"
            )

    try:
        if _stable_metadata(root_metadata) != _stable_metadata(os.fstat(root_fd)):
            raise PackageAuditError("package root identity changed while opening")
        walk(root_fd, "", 0)
        root_after = os.fstat(root_fd)
        if _stable_metadata(root_metadata) != _stable_metadata(root_after):
            raise PackageAuditError("package root identity changed during audit")
        verify_fd, verify_metadata = _open_root_without_symlinks(root_path)
        try:
            if _stable_metadata(root_after) != _stable_metadata(verify_metadata):
                raise PackageAuditError(
                    "package root or ancestor changed during audit"
                )
        finally:
            os.close(verify_fd)
    finally:
        os.close(root_fd)

    if _ATLAS_PATH not in sha256_by_path:
        findings.append(
            PackageFinding(_ATLAS_PATH, "missing_atlas", "fixed JHU atlas is required")
        )
    elif sha256_by_path[_ATLAS_PATH] != _ATLAS_SHA256:
        findings.append(
            PackageFinding(_ATLAS_PATH, "atlas_hash", "JHU atlas hash does not match")
        )

    files.sort()
    nifti_files.sort()
    cache_files.sort()
    log_files.sort()
    compiled_binaries.sort()
    executables.sort()
    findings.sort()
    manifest = hashlib.sha256()
    for relative in files:
        manifest.update(relative.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(sha256_by_path[relative].encode("ascii"))
        manifest.update(b"\n")

    return PackageAudit(
        root=".",
        files=files,
        nifti_files=nifti_files,
        forbidden_matches=findings,
        cache_files=cache_files,
        log_files=log_files,
        compiled_binaries=compiled_binaries,
        executables=executables,
        sha256_by_path={key: sha256_by_path[key] for key in files},
        total_bytes=total_bytes,
        manifest_sha256=manifest.hexdigest(),
    )
