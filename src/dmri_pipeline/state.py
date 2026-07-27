"""Durable, recoverable stage state for the dMRI pipeline."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

from .config import PipelineConfig


_STAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_NAME = ".stage_complete.json"


class StageStateError(RuntimeError):
    """Raised when stage state is unsafe, corrupt, or inconsistent."""


@dataclass(frozen=True)
class StageContext:
    """Immutable configuration and provenance shared by stage executions."""

    config: PipelineConfig
    package_root: Path
    subject_root: Path
    software: Mapping[str, str]

    def __post_init__(self) -> None:
        package_root = Path(self.package_root).expanduser().resolve(strict=False)
        subject_root = Path(self.subject_root).expanduser().resolve(strict=False)
        if not isinstance(self.config, PipelineConfig):
            raise StageStateError("stage context config must be a PipelineConfig")
        try:
            software = dict(sorted(self.software.items()))
        except (AttributeError, TypeError, ValueError) as error:
            raise StageStateError("software provenance must be a string mapping") from error
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in software.items()
        ):
            raise StageStateError("software provenance must be a string mapping")
        object.__setattr__(self, "package_root", package_root)
        object.__setattr__(self, "subject_root", subject_root)
        object.__setattr__(self, "software", MappingProxyType(software))


@dataclass(frozen=True)
class StageSpec:
    """Executable stage definition and regular files that determine currency.

    Source and resource dependencies must be explicit regular-file paths.
    Directory and symbolic-link dependencies are rejected because their entry
    identity cannot be represented safely by the compact stage signature.
    """

    name: str
    action: Callable[[Path], None]
    validator: Callable[[Path], Sequence[Path]]
    input_paths: tuple[Path, ...]
    source_paths: tuple[Path, ...]
    resource_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        _validate_stage_name(self.name)
        if not callable(self.action) or not callable(self.validator):
            raise StageStateError("stage action and validator must be callable")
        object.__setattr__(
            self, "input_paths", tuple(Path(path) for path in self.input_paths)
        )
        object.__setattr__(
            self, "source_paths", tuple(Path(path) for path in self.source_paths)
        )
        object.__setattr__(
            self, "resource_paths", tuple(Path(path) for path in self.resource_paths)
        )


@dataclass(frozen=True)
class StageOutcome:
    """Result of running or resuming a stage."""

    stage: str
    status: Literal["completed", "skipped"]
    directory: Path
    record_path: Path


@dataclass(frozen=True)
class StageRecord:
    """Validated, immutable representation of a completion record."""

    stage: str
    subject_id: str
    package_version: str
    config_sha256: str
    stage_signature: str
    started_utc: str
    completed_utc: str
    inputs: tuple[Mapping[str, object], ...]
    outputs: tuple[Mapping[str, object], ...]
    software: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_stage_name(self.stage)
        for label, value in (
            ("subject_id", self.subject_id),
            ("package_version", self.package_version),
            ("started_utc", self.started_utc),
            ("completed_utc", self.completed_utc),
        ):
            if not isinstance(value, str) or not value:
                raise StageStateError(f"completion record {label} must be non-empty")
        for label, value in (
            ("config_sha256", self.config_sha256),
            ("stage_signature", self.stage_signature),
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise StageStateError(f"completion record {label} is not a SHA-256")

        normalized_inputs = tuple(
            MappingProxyType(_validate_input_entry(entry)) for entry in self.inputs
        )
        normalized_outputs = tuple(
            MappingProxyType(_validate_output_entry(entry)) for entry in self.outputs
        )
        try:
            software = dict(sorted(self.software.items()))
        except (AttributeError, TypeError, ValueError) as error:
            raise StageStateError("completion record software must be a mapping") from error
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in software.items()
        ):
            raise StageStateError("completion record software must contain strings")
        object.__setattr__(
            self,
            "inputs",
            tuple(sorted(normalized_inputs, key=lambda entry: str(entry["path"]))),
        )
        object.__setattr__(
            self,
            "outputs",
            tuple(
                sorted(
                    normalized_outputs,
                    key=lambda entry: str(entry["relative_path"]),
                )
            ),
        )
        object.__setattr__(self, "software", MappingProxyType(software))

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic JSON shape stored on disk."""
        return {
            "stage": self.stage,
            "subject_id": self.subject_id,
            "package_version": self.package_version,
            "config_sha256": self.config_sha256,
            "stage_signature": self.stage_signature,
            "started_utc": self.started_utc,
            "completed_utc": self.completed_utc,
            "inputs": [dict(entry) for entry in self.inputs],
            "outputs": [dict(entry) for entry in self.outputs],
            "software": dict(self.software),
        }

    @classmethod
    def from_dict(cls, value: object) -> StageRecord:
        """Parse and strictly validate a completion-record JSON object."""
        if not isinstance(value, dict):
            raise StageStateError("completion record must be a JSON object")
        required = {
            "stage",
            "subject_id",
            "package_version",
            "config_sha256",
            "stage_signature",
            "started_utc",
            "completed_utc",
            "inputs",
            "outputs",
            "software",
        }
        if set(value) != required:
            raise StageStateError("completion record fields are missing or unexpected")
        inputs = value["inputs"]
        outputs = value["outputs"]
        software = value["software"]
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise StageStateError("completion record inputs and outputs must be lists")
        if not isinstance(software, dict):
            raise StageStateError("completion record software must be a mapping")
        try:
            return cls(
                stage=value["stage"],  # type: ignore[arg-type]
                subject_id=value["subject_id"],  # type: ignore[arg-type]
                package_version=value["package_version"],  # type: ignore[arg-type]
                config_sha256=value["config_sha256"],  # type: ignore[arg-type]
                stage_signature=value["stage_signature"],  # type: ignore[arg-type]
                started_utc=value["started_utc"],  # type: ignore[arg-type]
                completed_utc=value["completed_utc"],  # type: ignore[arg-type]
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                software=software,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, StageStateError):
                raise
            raise StageStateError("completion record has invalid values") from error


def stage_signature(
    config: PipelineConfig,
    stage_name: str,
    source_paths: Sequence[Path],
    resource_paths: Sequence[Path],
) -> str:
    """Hash stage identity, package version, config, sources, and resources.

    Source and resource paths must name regular files directly. Directories and
    paths containing symbolic-link components are rejected explicitly.
    """
    return _stage_signature(
        config,
        stage_name,
        source_paths,
        resource_paths,
        _package_version(Path(__file__).resolve().parents[2]),
    )


@dataclass(frozen=True)
class StageRunner:
    """Execute stages in work directories and atomically promote valid results."""

    context: StageContext

    def work_dir(self, stage_name: str) -> Path:
        """Return a containment-checked work directory for *stage_name*."""
        _validate_stage_name(stage_name)
        return self._subject_path(
            self.context.subject_root / ".work" / stage_name,
            purpose="work directory",
        )

    def final_dir(self, stage_name: str) -> Path:
        """Return a containment-checked final directory for *stage_name*."""
        _validate_stage_name(stage_name)
        return self._subject_path(
            self.context.subject_root / stage_name,
            purpose="final directory",
        )

    def record_path(self, stage_name: str) -> Path:
        """Return the completion-record path for *stage_name*."""
        return self.final_dir(stage_name) / _RECORD_NAME

    def is_current(self, spec: StageSpec) -> bool:
        """Return whether final state exactly matches the stage's dependencies."""
        final_dir = self.final_dir(spec.name)
        record_path = final_dir / _RECORD_NAME
        if not final_dir.is_dir() or not record_path.is_file():
            return False
        try:
            record = StageRecord.from_dict(
                json.loads(record_path.read_text(encoding="utf-8"))
            )
            if (
                record.stage != spec.name
                or record.subject_id != self.context.config.subject_id
                or record.package_version != self._package_version
                or record.config_sha256 != _config_sha256(self.context.config)
                or record.stage_signature != self._signature(spec)
                or record.software != self.context.software
            ):
                return False
            if record.inputs != self._input_records(spec.input_paths):
                return False
            for output in record.outputs:
                path = self._recorded_output_path(
                    final_dir, str(output["relative_path"])
                )
                if (
                    not path.is_file()
                    or path.stat().st_size != output["size"]
                    or _sha256_file(path, "output") != output["sha256"]
                ):
                    return False
        except (OSError, ValueError, TypeError, StageStateError):
            return False
        return True

    def run(self, spec: StageSpec) -> StageOutcome:
        """Run or resume *spec*, preserving recoverable state on every failure."""
        final_dir = self.final_dir(spec.name)
        record_path = final_dir / _RECORD_NAME
        if _entry_exists(final_dir):
            if self.is_current(spec):
                return StageOutcome(spec.name, "skipped", final_dir, record_path)
            raise StageStateError(
                f"Stage {spec.name!r} has a noncurrent final directory; "
                "force/invalidate it before rerunning"
            )

        work_dir = self.work_dir(spec.name)
        if work_dir.is_symlink():
            raise StageStateError(
                f"Stage {spec.name!r} work directory cannot be a symbolic link"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        if not work_dir.is_dir():
            raise StageStateError(f"Stage {spec.name!r} work path is not a directory")

        started_utc = _utc_now()
        inputs = self._input_records(spec.input_paths)
        signature = self._signature(spec)
        spec.action(work_dir)
        validated_paths = spec.validator(work_dir)
        outputs = self._validate_outputs(work_dir, validated_paths)
        if work_dir.is_symlink():
            raise StageStateError(
                f"Stage {spec.name!r} action replaced its work directory unsafely"
            )
        self._subject_path(work_dir, purpose="work directory")

        record = StageRecord(
            stage=spec.name,
            subject_id=self.context.config.subject_id,
            package_version=self._package_version,
            config_sha256=_config_sha256(self.context.config),
            stage_signature=signature,
            started_utc=started_utc,
            completed_utc=_utc_now(),
            inputs=inputs,
            outputs=outputs,
            software=self.context.software,
        )
        _write_record(record, work_dir / _RECORD_NAME)
        if _entry_exists(final_dir):
            raise StageStateError(
                f"Stage {spec.name!r} final directory appeared during execution; "
                "force/invalidate it before rerunning"
            )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            _rename_no_replace(work_dir, final_dir)
        except FileExistsError as error:
            raise StageStateError(
                f"Stage {spec.name!r} final directory appeared during promotion; "
                "it was preserved and must be force/invalidated before rerunning"
            ) from error
        except OSError as error:
            raise StageStateError(
                f"Cannot atomically promote stage {spec.name!r} without clobbering"
            ) from error
        return StageOutcome(spec.name, "completed", final_dir, record_path)

    def invalidate_from(
        self, stage_names: Sequence[str], start_stage: str
    ) -> None:
        """Move *start_stage* and all later stage state into a local archive."""
        names = tuple(stage_names)
        for name in names:
            _validate_stage_name(name)
        _validate_stage_name(start_stage)
        if len(set(names)) != len(names):
            raise StageStateError("stage order must not contain duplicate names")
        try:
            start = names.index(start_stage)
        except ValueError as error:
            raise StageStateError(
                f"start stage {start_stage!r} is not in the stage order"
            ) from error

        affected: list[tuple[str, Path, str]] = []
        for name in names[start:]:
            for kind, path in (
                ("final", self.final_dir(name)),
                ("work", self.work_dir(name)),
            ):
                if _entry_exists(path):
                    affected.append((kind, path, name))
        if not affected:
            return

        invalidated_root = self._subject_path(
            self.context.subject_root / ".invalidated",
            purpose="invalidation directory",
        )
        invalidated_root.mkdir(parents=True, exist_ok=True)
        archive = invalidated_root / _archive_name()
        suffix = 1
        while _entry_exists(archive):
            archive = invalidated_root / f"{_archive_name()}-{suffix:02d}"
            suffix += 1
        for kind in ("final", "work"):
            (archive / kind).mkdir(parents=True, exist_ok=False)
        for kind, source, name in affected:
            os.replace(source, archive / kind / name)

    @property
    def _package_version(self) -> str:
        return _package_version(self.context.package_root)

    def _signature(self, spec: StageSpec) -> str:
        return _stage_signature(
            self.context.config,
            spec.name,
            spec.source_paths,
            spec.resource_paths,
            self._package_version,
        )

    def _input_records(
        self, paths: Sequence[Path]
    ) -> tuple[Mapping[str, object], ...]:
        records = [
            {
                "path": _recorded_path(
                    path, self.context.config, self.context.package_root
                ),
                "sha256": _sha256_file(path, "input"),
            }
            for path in paths
        ]
        return tuple(
            MappingProxyType(record)
            for record in sorted(records, key=lambda item: item["path"])
        )

    def _validate_outputs(
        self, work_dir: Path, paths: Sequence[Path]
    ) -> tuple[Mapping[str, object], ...]:
        if paths is None or isinstance(paths, (str, bytes, Path)):
            raise StageStateError("stage validator must return a sequence of paths")
        outputs: list[dict[str, object]] = []
        seen: set[str] = set()
        for raw_path in paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = work_dir / candidate
            path = candidate.resolve(strict=False)
            try:
                relative = path.relative_to(work_dir.resolve(strict=False))
            except ValueError as error:
                raise StageStateError(
                    "stage validator returned an output outside its work directory"
                ) from error
            relative_path = relative.as_posix()
            if relative_path == _RECORD_NAME or relative_path in seen:
                raise StageStateError(
                    "stage validator returned a reserved or duplicate output"
                )
            if not path.is_file():
                raise StageStateError(
                    f"stage validator required output is missing: {relative_path}"
                )
            seen.add(relative_path)
            outputs.append(
                {
                    "relative_path": relative_path,
                    "sha256": _sha256_file(path, "output"),
                    "size": path.stat().st_size,
                }
            )
        return tuple(
            MappingProxyType(output)
            for output in sorted(outputs, key=lambda item: item["relative_path"])
        )

    def _recorded_output_path(self, final_dir: Path, relative_path: str) -> Path:
        candidate = (final_dir / relative_path).resolve(strict=False)
        try:
            candidate.relative_to(final_dir.resolve(strict=False))
        except ValueError as error:
            raise StageStateError(
                "completion record output escapes the final directory"
            ) from error
        return candidate

    def _subject_path(self, path: Path, *, purpose: str) -> Path:
        candidate = Path(os.path.abspath(os.fspath(path)))
        subject_root = self.context.subject_root
        try:
            relative = candidate.relative_to(subject_root)
        except ValueError as error:
            raise StageStateError(f"{purpose} points outside subject root") from error
        current = subject_root
        for index, part in enumerate(relative.parts):
            current /= part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                break
            except OSError as error:
                raise StageStateError(f"Cannot inspect {purpose}: {current}") from error
            if stat.S_ISLNK(metadata.st_mode):
                raise StageStateError(
                    f"{purpose} contains a symbolic link that could point "
                    "outside subject root"
                )
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise StageStateError(
                    f"{purpose} contains a non-directory path component: {current}"
                )
        return candidate


def _validate_stage_name(name: object) -> None:
    if not isinstance(name, str) or not _STAGE_NAME.fullmatch(name):
        raise StageStateError(
            "stage name must start with an alphanumeric character and contain "
            "only letters, digits, dots, underscores, or hyphens"
        )


def _validate_input_entry(entry: object) -> dict[str, object]:
    if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
        raise StageStateError("completion record input entry is invalid")
    path = entry["path"]
    sha256 = entry["sha256"]
    if not isinstance(path, str) or not path:
        raise StageStateError("completion record input path is invalid")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise StageStateError("completion record input SHA-256 is invalid")
    return {"path": path, "sha256": sha256}


def _validate_output_entry(entry: object) -> dict[str, object]:
    if not isinstance(entry, Mapping) or set(entry) != {
        "relative_path",
        "sha256",
        "size",
    }:
        raise StageStateError("completion record output entry is invalid")
    relative_path = entry["relative_path"]
    sha256 = entry["sha256"]
    size = entry["size"]
    if not isinstance(relative_path, str) or not relative_path:
        raise StageStateError("completion record output path is invalid")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or relative_path == _RECORD_NAME:
        raise StageStateError("completion record output path must be safe and relative")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise StageStateError("completion record output SHA-256 is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise StageStateError("completion record output size is invalid")
    return {"relative_path": path.as_posix(), "sha256": sha256, "size": size}


def _config_sha256(config: PipelineConfig) -> str:
    return hashlib.sha256(_canonical_json(config.canonical_dict())).hexdigest()


def _stage_signature(
    config: PipelineConfig,
    stage_name: str,
    source_paths: Sequence[Path],
    resource_paths: Sequence[Path],
    package_version: str,
) -> str:
    _validate_stage_name(stage_name)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "config": config.canonical_dict(),
                "package_version": package_version,
                "stage": stage_name,
            }
        )
    )
    for category, paths in (
        ("resource", resource_paths),
        ("source", source_paths),
    ):
        manifests = [_content_manifest(Path(path), category) for path in paths]
        for manifest in sorted(manifests, key=lambda item: str(item["path"])):
            digest.update(_canonical_json(manifest))
    return digest.hexdigest()


def _content_manifest(path: Path, category: str) -> dict[str, object]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    symlink = _first_symlink_component(lexical)
    if symlink is not None:
        raise StageStateError(
            f"{category} symbolic-link dependencies are unsupported: {symlink}"
        )
    try:
        metadata = lexical.lstat()
    except OSError as error:
        raise StageStateError(f"Cannot inspect {category} path: {path}") from error
    if stat.S_ISDIR(metadata.st_mode):
        raise StageStateError(
            f"{category} directory dependencies are unsupported; "
            "list each regular file explicitly"
        )
    if stat.S_ISREG(metadata.st_mode):
        return {
            "category": category,
            "path": lexical.as_posix(),
            "sha256": _sha256_file(lexical, category),
        }
    raise StageStateError(f"Cannot hash missing {category} path: {path}")


def _first_symlink_component(path: Path) -> Path | None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StageStateError(f"Cannot inspect dependency path: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            return current
    return None


def _recorded_path(
    path: Path, config: PipelineConfig, package_root: Path
) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    roots = (
        config.config_path.parent.resolve(strict=False),
        Path(package_root).resolve(strict=False),
    )
    for root in roots:
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise StageStateError(f"Cannot hash {label} file: {path}") from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _package_version(package_root: Path) -> str:
    version_path = Path(package_root) / "VERSION"
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise StageStateError(f"Cannot read package version: {version_path}") from error
    if not version:
        raise StageStateError(f"Package version is empty: {version_path}")
    return version


def _write_record(record: StageRecord, destination: Path) -> None:
    payload = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".stage-record-",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
        os.replace(temporary_name, destination)
    except OSError as error:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise StageStateError(
            f"Cannot write completion record: {destination}"
        ) from error


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination is absent."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    ctypes.set_errno(0)

    if sys.platform == "darwin":
        try:
            rename = libc.renamex_np
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(source_bytes, destination_bytes, 0x00000004)
        except AttributeError:
            try:
                rename_at = libc.renameatx_np
            except AttributeError as error:
                raise OSError(
                    errno.ENOTSUP,
                    "atomic no-replace rename is unavailable",
                    destination,
                ) from error
            rename_at.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename_at.restype = ctypes.c_int
            result = rename_at(
                -2, source_bytes, -2, destination_bytes, 0x00000004
            )
    elif sys.platform.startswith("linux"):
        try:
            rename_at = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable",
                destination,
            ) from error
        rename_at.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_at.restype = ctypes.c_int
        result = rename_at(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rename is unavailable on this platform",
            destination,
        )

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number, os.strerror(error_number), destination
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _archive_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
