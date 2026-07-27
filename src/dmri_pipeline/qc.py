"""Patient-generic, deterministic visual quality-control products.

The figures in this module are technical/research quicklooks.  They deliberately
record ``NOT_REVIEWED`` until a human has inspected them and never make a
clinical claim.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import math
import os
import secrets
import stat
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError, ImageDataError
from scipy import ndimage

from .state import StageContext
from .utils import round_shells


FIGURE_FILENAMES: Mapping[str, str] = MappingProxyType(
    OrderedDict(
        (
            ("input", "00_input_b0.png"),
            ("shell_distribution", "00_shell_distribution.png"),
            ("stripe", "00_pre_denoise_stripe_qc.png"),
            ("denoise_pa", "01_pca_pa.png"),
            ("denoise_ap", "01_pca_ap.png"),
            ("gibbs_pa", "02_gibbs_pa.png"),
            ("gibbs_ap", "02_gibbs_ap.png"),
            ("topup", "03_topup.png"),
            ("bet", "04_bet_mask.png"),
            ("eddy_images", "05_eddy_images.png"),
            ("eddy_motion", "05_eddy_motion.png"),
            ("dti", "06_dti.png"),
            ("dki", "07_dki.png"),
            ("dki_direct", "07_dki_direct.png"),
            ("noddi", "08_noddi.png"),
            ("jhu_48roi", "09_jhu_overlay.png"),
            ("overview", "${subject_id}_stepwise_overview.png"),
        )
    )
)
STAGE_FIGURES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    OrderedDict(
        (
            ("00_input_audit", ("input", "shell_distribution")),
            ("00_pre_denoise_motion_qc", ("stripe",)),
            ("01_denoise", ("denoise_pa", "denoise_ap")),
            ("02_gibbs", ("gibbs_pa", "gibbs_ap")),
            ("03_topup", ("topup",)),
            ("04_bet", ("bet",)),
            ("05_eddy", ("eddy_images", "eddy_motion")),
            ("06_dti", ("dti",)),
            ("07_dki", ("dki",)),
            ("07_dki_direct", ("dki_direct",)),
            ("08_noddi", ("noddi",)),
            ("09_jhu_48roi", ("jhu_48roi",)),
            ("overview", ("overview",)),
        )
    )
)
FIGURE_IDS = tuple(FIGURE_FILENAMES)
NON_OVERVIEW_FIGURE_IDS = FIGURE_IDS[:-1]
ACCEPTED_STAGE_NAMES = tuple(STAGE_FIGURES)

DTI_KEYS = ("FA", "MD", "AD", "RD")
DKI_KEYS = ("FA", "MD", "AD", "RD", "MK", "AK", "RK")
DKI_DIRECT_KEYS = ("MD", "MK", "S0")
NODDI_KEYS = ("ODI", "FICVF", "FISO")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_GRID_ATOL = 1e-5
_AMBIGUOUS = 1.15
_HIGH = 1.25


class QCError(ValueError):
    """Raised when a QC input, destination, or generated figure is unsafe."""


@dataclass(frozen=True)
class StageQCContext:
    """Explicit assignment of every scientific input needed by visual QC."""

    stage_context: StageContext
    output_directory: Path
    bvals: Path
    raw_pa: Path
    raw_ap: Path
    stripe_metrics_csv: Path
    stripe_decision_json: Path
    denoised_pa: Path
    denoised_ap: Path
    gibbs_pa: Path
    gibbs_ap: Path
    topup_merged_b0: Path
    topup_corrected_b0: Path
    topup_manifest_json: Path
    hifi_b0: Path
    brain_mask: Path
    eddy_dwi: Path
    eddy_parameters: Path
    eddy_movement_rms: Path
    eddy_outlier_map: Path
    dti_maps: Mapping[str, Path]
    dki_maps: Mapping[str, Path]
    dki_direct_maps: Mapping[str, Path]
    noddi_maps: Mapping[str, Path]
    warped_atlas: Path

    def __post_init__(self) -> None:
        if not isinstance(self.stage_context, StageContext):
            raise QCError("stage_context must be a StageContext")
        for field in (
            "output_directory",
            "bvals",
            "raw_pa",
            "raw_ap",
            "stripe_metrics_csv",
            "stripe_decision_json",
            "denoised_pa",
            "denoised_ap",
            "gibbs_pa",
            "gibbs_ap",
            "topup_merged_b0",
            "topup_corrected_b0",
            "topup_manifest_json",
            "hifi_b0",
            "brain_mask",
            "eddy_dwi",
            "eddy_parameters",
            "eddy_movement_rms",
            "eddy_outlier_map",
            "warped_atlas",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        object.__setattr__(
            self, "dti_maps", _exact_map(self.dti_maps, DTI_KEYS, "DTI")
        )
        object.__setattr__(
            self, "dki_maps", _exact_map(self.dki_maps, DKI_KEYS, "DKI")
        )
        object.__setattr__(
            self,
            "dki_direct_maps",
            _exact_map(self.dki_direct_maps, DKI_DIRECT_KEYS, "direct DKI"),
        )
        object.__setattr__(
            self, "noddi_maps", _exact_map(self.noddi_maps, NODDI_KEYS, "NODDI")
        )
        _validate_path_assignment(self)


@dataclass(frozen=True)
class _HeldInput:
    path: Path
    descriptor: int
    signature: tuple[int, int, int, int, int, int]
    digest: str
    snapshot_path: Path
    snapshot_descriptor: int
    snapshot_signature: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _OwnedFile:
    name: str
    descriptor: int
    identity: tuple[int, int]


@dataclass
class _PendingCommitFile:
    name: str
    descriptor: int
    identity: tuple[int, int] | None = None


class _OwnedWriteHandle:
    """File-like staging target retaining a nominal path only for diagnostics."""

    def __init__(self, stream: BinaryIO, path: Path) -> None:
        self._stream = stream
        self.path = path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def parent(self) -> Path:
        return self.path.parent

    def write(self, content: bytes) -> int:
        return self._stream.write(content)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def writable(self) -> bool:
        return self._stream.writable()

    def readable(self) -> bool:
        return self._stream.readable()

    def seekable(self) -> bool:
        return self._stream.seekable()

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


class _DirectoryAnchor:
    """Pin one existing directory and its nominal pathname binding."""

    def __init__(
        self,
        path: Path,
        parent_descriptor: int,
        root_descriptor: int,
        identity: tuple[int, int],
        *,
        label: str,
    ) -> None:
        self.path = _absolute(path)
        self.name = self.path.name
        self.parent_descriptor = parent_descriptor
        self.root_descriptor = root_descriptor
        self.identity = identity
        self.label = label
        self._closed = False

    @classmethod
    def bind(
        cls,
        path: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
        label: str,
    ) -> "_DirectoryAnchor":
        absolute = _absolute(path)
        if not absolute.name:
            raise QCError(f"{label} must not be a filesystem root")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_descriptor: int | None = None
        root_descriptor: int | None = None
        try:
            parent_descriptor = os.open(absolute.parent, flags)
            named = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            named_identity = (int(named.st_dev), int(named.st_ino))
            if (
                not stat.S_ISDIR(named.st_mode)
                or (
                    expected_identity is not None
                    and named_identity != expected_identity
                )
            ):
                raise QCError(f"{label} binding changed before it was pinned")
            root_descriptor = os.open(
                absolute.name,
                flags,
                dir_fd=parent_descriptor,
            )
            root = os.fstat(root_descriptor)
            root_identity = (int(root.st_dev), int(root.st_ino))
            if (
                not stat.S_ISDIR(root.st_mode)
                or root_identity != named_identity
            ):
                raise QCError(f"{label} binding changed before it was pinned")
            return cls(
                absolute,
                parent_descriptor,
                root_descriptor,
                root_identity,
                label=label,
            )
        except BaseException:
            if root_descriptor is not None:
                try:
                    os.close(root_descriptor)
                except OSError:
                    pass
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
            raise

    def verify_nominal_binding(self) -> None:
        if self._closed:
            raise QCError(f"{self.label} anchor is closed")
        try:
            root = os.fstat(self.root_descriptor)
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            nominal_descriptor = os.open(self.path, flags)
        except OSError as error:
            raise QCError(f"{self.label} was replaced during use") from error
        try:
            identities = {
                (int(root.st_dev), int(root.st_ino)),
                (int(named.st_dev), int(named.st_ino)),
                _fd_identity(nominal_descriptor),
            }
            if (
                not stat.S_ISDIR(root.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or identities != {self.identity}
            ):
                raise QCError(f"{self.label} was replaced during use")
        finally:
            os.close(nominal_descriptor)

    def close(self) -> None:
        if self._closed:
            return
        for descriptor in (self.root_descriptor, self.parent_descriptor):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._closed = True

    def __enter__(self) -> "_DirectoryAnchor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _OwnedDirectory:
    """Flat private directory whose root and files are owned by exact inode."""

    def __init__(
        self,
        anchor: _DirectoryAnchor,
        files: dict[str, _OwnedFile],
        pending: dict[int, str],
    ) -> None:
        self._anchor = anchor
        self._files = files
        self._pending = pending
        self._sealed_signature: tuple[int, int, int, int, int, int] | None = None
        self._closed = False

    @property
    def path(self) -> Path:
        return self._anchor.path

    @property
    def root_descriptor(self) -> int:
        return self._anchor.root_descriptor

    @property
    def parent_descriptor(self) -> int:
        return self._anchor.parent_descriptor

    @property
    def identity(self) -> tuple[int, int]:
        return self._anchor.identity

    @classmethod
    def create(
        cls,
        parent: Path,
        *,
        prefix: str,
        names: Sequence[str] = (),
    ) -> "_OwnedDirectory":
        parent_path = Path(os.path.realpath(parent))
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_descriptor: int | None = None
        root_descriptor: int | None = None
        root_identity: tuple[int, int] | None = None
        root_name: str | None = None
        try:
            parent_descriptor = os.open(parent_path, flags)
            for _attempt in range(128):
                candidate = f"{prefix}{secrets.token_hex(16)}"
                try:
                    os.mkdir(
                        candidate,
                        mode=0o700,
                        dir_fd=parent_descriptor,
                    )
                    root_name = candidate
                    break
                except FileExistsError:
                    continue
            if root_name is None:
                raise QCError("cannot allocate a unique private directory")
            root_descriptor = os.open(
                root_name,
                flags,
                dir_fd=parent_descriptor,
            )
            root = os.fstat(root_descriptor)
            if not stat.S_ISDIR(root.st_mode):
                raise QCError("private directory root is not a directory")
            root_identity = (int(root.st_dev), int(root.st_ino))
            named = os.stat(
                root_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(named.st_mode)
                or (int(named.st_dev), int(named.st_ino)) != root_identity
            ):
                raise QCError("private directory root identity changed")
            anchor = _DirectoryAnchor(
                parent_path / root_name,
                parent_descriptor,
                root_descriptor,
                root_identity,
                label="private temporary directory binding",
            )
            parent_descriptor = None
            root_descriptor = None
            owner = cls(anchor, {}, {})
            try:
                for name in names:
                    owner.reserve(name)
                owner.seal()
                return owner
            except BaseException as error:
                cleanup_error = owner.close(preserve_primary=True)
                if cleanup_error is not None:
                    if hasattr(error, "add_note"):
                        error.add_note(str(cleanup_error))
                    owner._abandon_failed_create()
                raise
        except BaseException as error:
            cleanup_error: BaseException | None = None
            if parent_descriptor is not None:
                if root_identity is None and root_descriptor is not None:
                    try:
                        recovered = _recover_descriptor_stat(
                            root_descriptor
                        )
                        if not stat.S_ISDIR(recovered.st_mode):
                            raise QCError(
                                "private directory cleanup root is not a directory"
                            )
                        root_identity = (
                            int(recovered.st_dev),
                            int(recovered.st_ino),
                        )
                    except BaseException as cleanup:
                        cleanup_error = cleanup
                if root_identity is not None:
                    try:
                        _remove_exact_empty_directory(
                            parent_descriptor,
                            root_identity,
                            root_descriptor=root_descriptor,
                        )
                    except BaseException as cleanup:
                        cleanup_error = cleanup
                if root_descriptor is not None:
                    try:
                        os.close(root_descriptor)
                    except OSError:
                        pass
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
            if cleanup_error is not None and hasattr(error, "add_note"):
                error.add_note(
                    f"private directory cleanup failed: {cleanup_error}"
                )
            raise

    def reserve(self, name: str, *, mode: int = 0o600) -> Path:
        _require_safe_basename(name, "private temporary file")
        if name in self._files:
            raise QCError("private temporary file names must be unique")
        descriptor: int | None = None
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                name,
                flags,
                mode,
                dir_fd=self.root_descriptor,
            )
            self._pending[descriptor] = name
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise QCError("private temporary output is not a regular file")
            identity = (int(details.st_dev), int(details.st_ino))
            self._files[name] = _OwnedFile(name, descriptor, identity)
            self._pending.pop(descriptor, None)
            return self.path / name
        except BaseException:
            if descriptor is not None and descriptor not in self._pending:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise

    def seal(self) -> None:
        os.fchmod(self.root_descriptor, 0o500)
        self._sealed_signature = _stat_signature(
            os.fstat(self.root_descriptor)
        )

    def descriptor(self, name: str) -> int:
        try:
            return self._files[name].descriptor
        except KeyError as error:
            raise QCError("undeclared private temporary file") from error

    def file_identity(self, name: str) -> tuple[int, int]:
        try:
            return self._files[name].identity
        except KeyError as error:
            raise QCError("undeclared private temporary file") from error

    @contextmanager
    def writer(self, name: str) -> Iterator[_OwnedWriteHandle]:
        self.verify()
        try:
            owned = self._files[name]
        except KeyError as error:
            raise QCError("undeclared private temporary file") from error
        os.ftruncate(owned.descriptor, 0)
        os.lseek(owned.descriptor, 0, os.SEEK_SET)
        duplicate = os.dup(owned.descriptor)
        try:
            stream = os.fdopen(duplicate, "w+b")
        except BaseException:
            try:
                os.close(duplicate)
            except OSError:
                pass
            raise
        handle = _OwnedWriteHandle(stream, self.path / name)
        try:
            yield handle
        except BaseException as primary:
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        f"staging output flush failed: {cleanup}"
                    )
            try:
                stream.close()
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        f"staging output close failed: {cleanup}"
                    )
            raise
        else:
            try:
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                stream.close()
        finally:
            os.lseek(owned.descriptor, 0, os.SEEK_SET)

    def write_text(self, name: str, text: str) -> None:
        try:
            content = text.encode("utf-8")
        except UnicodeError as error:
            raise QCError("cannot encode private temporary text") from error
        with self.writer(name) as stream:
            stream.write(content)

    def read_bytes(self, name: str) -> bytes:
        descriptor = self.descriptor(name)
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            content.extend(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return bytes(content)

    def verify(self) -> None:
        self._anchor.verify_nominal_binding()
        if self._sealed_signature is None:
            raise QCError("private temporary directory is not sealed")
        if (
            _stat_signature(os.fstat(self.root_descriptor))
            != self._sealed_signature
        ):
            raise QCError("private temporary directory changed")
        try:
            names = os.listdir(self.root_descriptor)
        except OSError as error:
            raise QCError("cannot list private temporary directory") from error
        if set(names) != set(self._files):
            raise QCError("private temporary directory contents changed")
        for name, owned in self._files.items():
            descriptor_details = os.fstat(owned.descriptor)
            named = os.stat(
                name,
                dir_fd=self.root_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(descriptor_details.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or descriptor_details.st_nlink != 1
                or named.st_nlink != 1
                or (
                    int(descriptor_details.st_dev),
                    int(descriptor_details.st_ino),
                )
                != owned.identity
                or (int(named.st_dev), int(named.st_ino)) != owned.identity
            ):
                raise QCError("private temporary output identity changed")

    def close(self, *, preserve_primary: bool = False) -> QCError | None:
        if self._closed:
            return None
        cleanup_error: QCError | None = None
        try:
            self._cleanup()
        except QCError as error:
            cleanup_error = error
        except BaseException as error:
            cleanup_error = QCError("private temporary cleanup failed")
            cleanup_error.__cause__ = error
        if cleanup_error is not None:
            if preserve_primary:
                return cleanup_error
            raise cleanup_error
        self._release()
        return None

    def _cleanup(self) -> None:
        os.fchmod(self.root_descriptor, 0o700)
        owned_identities = {owned.identity for owned in self._files.values()}
        descriptor_identities: dict[int, tuple[int, int]] = {}
        for owned in self._files.values():
            details = os.fstat(owned.descriptor)
            identity = (int(details.st_dev), int(details.st_ino))
            if not stat.S_ISREG(details.st_mode) or identity != owned.identity:
                raise QCError("private temporary cleanup identity changed")
            descriptor_identities[owned.descriptor] = identity
        for descriptor in self._pending:
            details = _recover_descriptor_stat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise QCError("private temporary pending output changed")
            identity = (int(details.st_dev), int(details.st_ino))
            descriptor_identities[descriptor] = identity
            owned_identities.add(identity)

        for descriptor in descriptor_identities:
            try:
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
            except OSError:
                pass

        remaining_owned: list[str] = []
        last_errors: list[OSError] = []
        for _attempt in range(3):
            last_errors = []
            for name in os.listdir(self.root_descriptor):
                try:
                    details = os.stat(
                        name,
                        dir_fd=self.root_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    last_errors.append(error)
                    continue
                identity = (int(details.st_dev), int(details.st_ino))
                if (
                    not stat.S_ISREG(details.st_mode)
                    or identity not in owned_identities
                ):
                    continue
                try:
                    checked = os.stat(
                        name,
                        dir_fd=self.root_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISREG(checked.st_mode)
                        and (int(checked.st_dev), int(checked.st_ino))
                        == identity
                    ):
                        os.unlink(name, dir_fd=self.root_descriptor)
                except OSError as error:
                    last_errors.append(error)
            remaining_owned = []
            for name in os.listdir(self.root_descriptor):
                try:
                    details = os.stat(
                        name,
                        dir_fd=self.root_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    last_errors.append(error)
                    continue
                if (
                    stat.S_ISREG(details.st_mode)
                    and (int(details.st_dev), int(details.st_ino))
                    in owned_identities
                ):
                    remaining_owned.append(name)
            if not remaining_owned and not last_errors:
                break
        unsafe = []
        for descriptor in descriptor_identities:
            details = _recover_descriptor_stat(descriptor)
            if details.st_nlink > 0 and details.st_size > 0:
                unsafe.append(descriptor)
        if remaining_owned or last_errors or unsafe:
            raise QCError(
                "private temporary cleanup is incomplete; retry is required"
            )

        remaining = os.listdir(self.root_descriptor)
        if remaining:
            return
        removed = _remove_exact_empty_directory(
            self.parent_descriptor,
            self.identity,
            root_descriptor=self.root_descriptor,
        )
        if not removed:
            raise QCError(
                "private temporary cleanup could not remove its root"
            )

    def _release(self) -> None:
        descriptors = {
            *(owned.descriptor for owned in self._files.values()),
            *self._pending,
        }
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._files.clear()
        self._pending.clear()
        self._anchor.close()
        self._closed = True

    def _abandon_failed_create(self) -> None:
        """Erase reachable bytes and close FDs if create-time cleanup fails."""
        descriptors = {
            *(owned.descriptor for owned in self._files.values()),
            *self._pending,
        }
        for descriptor in descriptors:
            try:
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._files.clear()
        self._pending.clear()
        self._anchor.close()
        self._closed = True

    def __enter__(self) -> "_OwnedDirectory":
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        exception: BaseException | None,
        _traceback: object,
    ) -> None:
        cleanup_error = self.close(preserve_primary=exception is not None)
        if (
            cleanup_error is not None
            and exception is not None
            and hasattr(exception, "add_note")
        ):
            exception.add_note(str(cleanup_error))


class _InputGuard:
    """Bind declared inputs to immutable, byte-exact private snapshots."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.declared_paths = tuple(paths)
        self.held: list[_HeldInput] = []
        self.paths: Mapping[Path, Path] = MappingProxyType({})
        self._snapshot_owner: _OwnedDirectory | None = None
        self._snapshot_root: Path | None = None
        self._snapshot_root_name: str | None = None
        self._snapshot_root_identity: tuple[int, int] | None = None
        self._snapshot_parent_descriptor: int | None = None
        self._snapshot_directory_descriptor: int | None = None
        self._snapshot_directory_signature: (
            tuple[int, int, int, int, int, int] | None
        ) = None
        self._owned_snapshot_files: dict[str, tuple[int, int]] = {}
        self._pending_snapshot_descriptors: dict[
            int, tuple[str, tuple[int, int]]
        ] = {}

    def __enter__(self) -> "_InputGuard":
        identities: dict[tuple[int, int], Path] = {}
        snapshots: dict[Path, Path] = {}
        snapshot_names = []
        for index, path in enumerate(self.declared_paths):
            suffix = "".join(path.suffixes) or ".bin"
            snapshot_names.append(f"{index:03d}{suffix}")
        self._snapshot_owner = _OwnedDirectory.create(
            Path(tempfile.gettempdir()),
            prefix="dmri-qc-inputs-",
            names=snapshot_names,
        )
        snapshot_root = self._snapshot_owner.path
        self._snapshot_root = snapshot_root
        self._snapshot_root_name = snapshot_root.name
        self._snapshot_parent_descriptor = (
            self._snapshot_owner.parent_descriptor
        )
        self._snapshot_directory_descriptor = (
            self._snapshot_owner.root_descriptor
        )
        self._snapshot_root_identity = self._snapshot_owner.identity
        self._owned_snapshot_files = {
            name: self._snapshot_owner.file_identity(name)
            for name in snapshot_names
        }
        try:
            for index, path in enumerate(self.declared_paths):
                descriptor: int | None = None
                snapshot_descriptor: int | None = None
                snapshot_identity: tuple[int, int] | None = None
                snapshot_name = snapshot_names[index]
                try:
                    descriptor = _open_regular_read(path)
                    metadata = os.fstat(descriptor)
                    identity = (int(metadata.st_dev), int(metadata.st_ino))
                    if identity in identities:
                        raise QCError(
                            "QC inputs must not be hard-link aliases: "
                            f"{path.name}"
                    )
                    identities[identity] = path
                    signature = _stat_signature(metadata)
                    snapshot_path = snapshot_root / snapshot_name
                    (
                        digest,
                        snapshot_descriptor,
                        snapshot_identity,
                    ) = _copy_descriptor_snapshot(
                        descriptor,
                        self._snapshot_directory_descriptor,
                        snapshot_name,
                        expected_identity=self._snapshot_owner.file_identity(
                            snapshot_name
                        ),
                    )
                    os.fchmod(snapshot_descriptor, 0o400)
                    snapshot_details = os.fstat(snapshot_descriptor)
                    if (
                        int(snapshot_details.st_dev),
                        int(snapshot_details.st_ino),
                    ) != snapshot_identity:
                        raise QCError(
                            "immutable QC input snapshot identity changed"
                        )
                    snapshot_signature = _stat_signature(snapshot_details)
                    if _hash_descriptor(snapshot_descriptor) != digest:
                        raise QCError(
                            "immutable QC input snapshot digest mismatch"
                        )
                    self.held.append(
                        _HeldInput(
                            path,
                            descriptor,
                            signature,
                            digest,
                            snapshot_path,
                            snapshot_descriptor,
                            snapshot_signature,
                        )
                    )
                    descriptor = None
                    snapshot_descriptor = None
                    snapshots[path] = snapshot_path
                except BaseException:
                    if snapshot_descriptor is not None:
                        _discard_snapshot_descriptor(
                            snapshot_descriptor,
                            self._snapshot_directory_descriptor,
                            snapshot_name,
                            snapshot_identity,
                        )
                    if descriptor is not None:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                    raise
            self._snapshot_directory_signature = _stat_signature(
                os.fstat(self._snapshot_directory_descriptor)
            )
            self.paths = MappingProxyType(snapshots)
            self.verify()
            return self
        except BaseException as error:
            cleanup_error = self.close(preserve_primary=True)
            if cleanup_error is not None and hasattr(error, "add_note"):
                error.add_note(str(cleanup_error))
            raise

    def verify(self) -> None:
        self._verify_snapshot_directory()
        for held in self.held:
            try:
                current = _open_regular_read(held.path)
            except QCError as error:
                raise QCError(
                    "QC input was replaced or changed during rendering"
                ) from error
            try:
                now = os.fstat(current)
                if _stat_signature(now) != held.signature:
                    raise QCError(
                        "QC input was replaced or changed during rendering"
                    )
                if _hash_descriptor(current) != held.digest:
                    raise QCError("QC input changed during rendering")
            finally:
                os.close(current)
            metadata = os.fstat(held.descriptor)
            if _stat_signature(metadata) != held.signature:
                raise QCError("QC input identity, link count, or metadata changed")
            if _hash_descriptor(held.descriptor) != held.digest:
                raise QCError("QC input changed during rendering")
            snapshot = os.fstat(held.snapshot_descriptor)
            if _stat_signature(snapshot) != held.snapshot_signature:
                raise QCError("immutable QC input snapshot changed")
            if _hash_descriptor(held.snapshot_descriptor) != held.digest:
                raise QCError("immutable QC input snapshot changed")
            try:
                current_snapshot = _open_regular_read(held.snapshot_path)
            except QCError as error:
                raise QCError("immutable QC input snapshot changed") from error
            try:
                if (
                    _stat_signature(os.fstat(current_snapshot))
                    != held.snapshot_signature
                    or _hash_descriptor(current_snapshot) != held.digest
                ):
                    raise QCError("immutable QC input snapshot changed")
            finally:
                os.close(current_snapshot)

    def _verify_snapshot_directory(self) -> None:
        if (
            self._snapshot_owner is None
            or
            self._snapshot_root is None
            or self._snapshot_root_name is None
            or self._snapshot_parent_descriptor is None
            or self._snapshot_directory_descriptor is None
            or self._snapshot_directory_signature is None
        ):
            raise QCError("immutable QC snapshot directory is unavailable")
        try:
            self._snapshot_owner.verify()
        except (OSError, QCError) as error:
            raise QCError("immutable QC snapshot directory changed")
        if (
            _stat_signature(os.fstat(self._snapshot_directory_descriptor))
            != self._snapshot_directory_signature
        ):
            raise QCError("immutable QC snapshot directory changed")

    def path(self, original: Path) -> Path:
        try:
            return self.paths[Path(original)]
        except KeyError as error:
            raise QCError("undeclared input has no immutable snapshot") from error

    def close(self, *, preserve_primary: bool = False) -> QCError | None:
        cleanup_error: QCError | None = None
        try:
            self._cleanup_owned_snapshots()
        except QCError as error:
            cleanup_error = error
        except Exception as error:
            cleanup_error = QCError("immutable snapshot cleanup failed")
            cleanup_error.__cause__ = error
        if cleanup_error is not None:
            if preserve_primary:
                return cleanup_error
            raise cleanup_error
        self._release_resources()
        return None

    def _release_resources(self) -> None:
        closed: set[int] = set()
        for held in self.held:
            for descriptor in (held.descriptor, held.snapshot_descriptor):
                if descriptor in closed:
                    continue
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                closed.add(descriptor)
        for descriptor in self._pending_snapshot_descriptors:
            if descriptor in closed:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
            closed.add(descriptor)
        self.held.clear()
        self.paths = MappingProxyType({})
        self._snapshot_owner = None
        self._snapshot_directory_descriptor = None
        self._snapshot_parent_descriptor = None
        self._snapshot_directory_signature = None
        self._snapshot_root_identity = None
        self._snapshot_root_name = None
        self._snapshot_root = None
        self._owned_snapshot_files.clear()
        self._pending_snapshot_descriptors.clear()

    def _cleanup_owned_snapshots(self) -> None:
        owner = self._snapshot_owner
        if owner is None:
            if not self.held and not self._pending_snapshot_descriptors:
                return
            raise QCError("immutable snapshot cleanup anchors are unavailable")
        cleanup_error = owner.close(preserve_primary=True)
        if cleanup_error is not None:
            raise QCError(
                "immutable snapshot cleanup is incomplete; retry is required"
            ) from cleanup_error

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        _traceback: object,
    ) -> None:
        cleanup_error = self.close(preserve_primary=exception is not None)
        if (
            cleanup_error is not None
            and exception is not None
            and hasattr(exception, "add_note")
        ):
            exception.add_note(str(cleanup_error))


@dataclass(frozen=True)
class _QCInputs:
    bvals: np.ndarray
    raw_pa: nib.spatialimages.SpatialImage
    raw_ap: nib.spatialimages.SpatialImage
    brain: np.ndarray
    affine: np.ndarray
    topup_manifest: Mapping[str, object]
    stripe_rows: tuple[Mapping[str, object], ...]
    stripe_decision: Mapping[str, object]
    eddy_parameters: np.ndarray
    movement_rms: np.ndarray
    outlier_map: np.ndarray
    snapshots: Mapping[Path, Path]


def generate_stage_qc(context: StageQCContext, stage_name: str) -> list[Path]:
    """Generate only one accepted stage's assigned QC figures."""
    _require_context(context)
    if stage_name not in STAGE_FIGURES:
        raise QCError(f"unknown QC stage name: {stage_name!r}")
    if stage_name == "overview":
        return [generate_overview(context)]
    destination = _require_empty_destination(context.output_directory)
    figure_ids = STAGE_FIGURES[stage_name]
    names = [_figure_basename(context, value) for value in figure_ids]
    try:
        with _InputGuard(_context_inputs(context)) as guard:
            inputs = _validate_scientific_inputs(context, guard)
            with _private_temp(
                context.output_directory.parent,
                names=names,
            ) as temporary:
                rendered = _render_figures(
                    context,
                    inputs,
                    figure_ids,
                    temporary,
                )
                guard.verify()
                temporary.verify()
                _commit_files(
                    destination,
                    rendered,
                    require_empty=True,
                    verifier=guard.verify,
                    source_owner=temporary,
                )
    finally:
        destination.close()
    return [_figure_path(context, figure_id) for figure_id in figure_ids]


def generate_overview(context: StageQCContext) -> Path:
    """Create the overview from exactly the 16 validated stage quicklooks."""
    _require_context(context)
    destination = _require_overview_prerequisites(context)
    name = _figure_basename(context, "overview")
    try:
        with _InputGuard(
            [*_context_inputs(context), *_non_overview_paths(context)]
        ) as guard:
            _validate_scientific_inputs(context, guard)
            with _private_temp(
                context.output_directory.parent,
                names=(name,),
            ) as temporary:
                target = temporary.path / name
                temporary.verify()
                with temporary.writer(name) as output:
                    _render_overview(
                        [
                            guard.path(path)
                            for path in _non_overview_paths(context)
                        ],
                        output,
                    )
                temporary.verify()
                _validate_owned_png(temporary, name)
                guard.verify()
                _commit_files(
                    destination,
                    [(target, target.name)],
                    verifier=guard.verify,
                    source_owner=temporary,
                )
    finally:
        destination.close()
    return _figure_path(context, "overview")


def generate_all_qc(context: StageQCContext) -> Mapping[str, Path]:
    """Atomically create the exact 17-figure manifest and QC JSON manifest."""
    _require_context(context)
    destination = _require_empty_destination(context.output_directory)
    names = [
        *(
            _figure_basename(context, figure_id)
            for figure_id in FIGURE_IDS
        ),
        "qc_manifest.json",
    ]
    try:
        with _InputGuard(_context_inputs(context)) as guard:
            inputs = _validate_scientific_inputs(context, guard)
            with _private_temp(
                context.output_directory.parent,
                names=names,
            ) as temporary:
                rendered = _render_figures(
                    context,
                    inputs,
                    NON_OVERVIEW_FIGURE_IDS,
                    temporary,
                )
                overview = (
                    temporary.path
                    / _figure_basename(context, "overview")
                )
                temporary.verify()
                with temporary.writer(overview.name) as output:
                    _render_overview(
                        [path for path, _ in rendered],
                        output,
                        source_owner=temporary,
                    )
                temporary.verify()
                _validate_owned_png(temporary, overview.name)
                rendered.append((overview, overview.name))
                manifest = temporary.path / "qc_manifest.json"
                _write_qc_manifest(
                    context,
                    rendered,
                    temporary,
                    manifest.name,
                )
                temporary.verify()
                guard.verify()
                _commit_files(
                    destination,
                    [*rendered, (manifest, manifest.name)],
                    require_empty=True,
                    verifier=guard.verify,
                    source_owner=temporary,
                )
    finally:
        destination.close()
    return MappingProxyType(
        OrderedDict(
            (figure_id, _figure_path(context, figure_id))
            for figure_id in FIGURE_IDS
        )
    )


def _exact_map(
    supplied: Mapping[str, Path], expected: tuple[str, ...], label: str
) -> Mapping[str, Path]:
    try:
        values = dict(supplied)
    except (TypeError, ValueError) as error:
        raise QCError(f"{label} maps must be a path mapping") from error
    if set(values) != set(expected):
        raise QCError(f"{label} maps must contain exact keys {list(expected)}")
    return MappingProxyType(OrderedDict((key, Path(values[key])) for key in expected))


def _require_context(context: object) -> StageQCContext:
    if not isinstance(context, StageQCContext):
        raise QCError("context must be a StageQCContext")
    _validate_path_assignment(context)
    return context


def _validate_path_assignment(context: StageQCContext) -> None:
    config = context.stage_context.config
    subject_root = context.stage_context.subject_root
    if subject_root.resolve(strict=False) != config.subject_output.resolve(strict=False):
        raise QCError("StageContext subject_root must equal configured subject output")
    if _absolute(context.raw_pa) != _absolute(config.dwi_pa):
        raise QCError("raw_pa must match the configured PA input exactly")
    if _absolute(context.raw_ap) != _absolute(config.b0_ap):
        raise QCError("raw_ap must match the configured AP input exactly")
    if _absolute(context.bvals) != _absolute(config.bvals):
        raise QCError("bvals must match the configured b-values exactly")

    raw = {_absolute(context.raw_pa), _absolute(context.raw_ap), _absolute(context.bvals)}
    for path in _context_inputs(context):
        _reject_traversal(path)
        _reject_symlink_components(path)
        if _absolute(path) not in raw:
            _relative_subject(path, subject_root)
    output = context.output_directory
    _reject_traversal(output)
    _reject_symlink_components(output)
    _relative_subject(output, subject_root, require_exists=False)
    if _absolute(output) in {_absolute(path) for path in _context_inputs(context)}:
        raise QCError("QC destination must not alias an input")
    if os.path.lexists(output):
        mode = output.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise QCError("QC destination must be a real directory")


def _context_inputs(context: StageQCContext) -> list[Path]:
    return [
        context.bvals,
        context.raw_pa,
        context.raw_ap,
        context.stripe_metrics_csv,
        context.stripe_decision_json,
        context.denoised_pa,
        context.denoised_ap,
        context.gibbs_pa,
        context.gibbs_ap,
        context.topup_merged_b0,
        context.topup_corrected_b0,
        context.topup_manifest_json,
        context.hifi_b0,
        context.brain_mask,
        context.eddy_dwi,
        context.eddy_parameters,
        context.eddy_movement_rms,
        context.eddy_outlier_map,
        *context.dti_maps.values(),
        *context.dki_maps.values(),
        *context.dki_direct_maps.values(),
        *context.noddi_maps.values(),
        context.warped_atlas,
    ]


def _validate_scientific_inputs(
    context: StageQCContext, snapshots: _InputGuard
) -> _QCInputs:
    frozen = snapshots.path
    raw_pa = _load_nifti(frozen(context.raw_pa), "raw PA", dimensions=(4,))
    raw_ap = _load_nifti(frozen(context.raw_ap), "raw AP", dimensions=(3, 4))
    _require_finite_image(raw_pa, "raw PA")
    _require_finite_image(raw_ap, "raw AP")
    _require_same_spatial_grid(raw_pa, raw_ap, "raw AP")
    bvals = _load_numeric_table(frozen(context.bvals), "b-values").reshape(-1)
    if bvals.size != raw_pa.shape[3] or not np.isfinite(bvals).all():
        raise QCError("b-value count must equal finite raw PA volume count")
    if np.any(bvals < 0):
        raise QCError("b-values must be nonnegative")
    if not np.any(bvals < 50.0):
        raise QCError("raw PA must contain at least one b0 with b < 50")

    diffusion_images: list[tuple[str, Path, tuple[int, ...]]] = [
        ("denoised PA", context.denoised_pa, (4,)),
        ("denoised AP", context.denoised_ap, (4,)),
        ("Gibbs PA", context.gibbs_pa, (4,)),
        ("Gibbs AP", context.gibbs_ap, (4,)),
        ("TOPUP merged b0", context.topup_merged_b0, (4,)),
        ("TOPUP corrected b0", context.topup_corrected_b0, (4,)),
        ("hifi b0", context.hifi_b0, (3,)),
        ("brain mask", context.brain_mask, (3,)),
        ("EDDY DWI", context.eddy_dwi, (4,)),
        ("warped atlas", context.warped_atlas, (3,)),
    ]
    loaded: dict[str, nib.spatialimages.SpatialImage] = {}
    for label, path, dimensions in diffusion_images:
        image = _load_nifti(frozen(path), label, dimensions=dimensions)
        _require_same_spatial_grid(raw_pa, image, label)
        _require_finite_image(image, label)
        loaded[label] = image
    for family, maps in (
        ("DTI", context.dti_maps),
        ("DKI", context.dki_maps),
        ("direct DKI", context.dki_direct_maps),
        ("NODDI", context.noddi_maps),
    ):
        for key, path in maps.items():
            image = _load_nifti(frozen(path), f"{family} {key}", dimensions=(3,))
            _require_same_spatial_grid(raw_pa, image, f"{family} {key}")
            _require_finite_image(image, f"{family} {key}")

    if loaded["denoised PA"].shape != raw_pa.shape:
        raise QCError("denoised PA shape must match raw PA")
    if loaded["Gibbs PA"].shape != raw_pa.shape:
        raise QCError("Gibbs PA shape must match raw PA")
    ap_count = 1 if len(raw_ap.shape) == 3 else raw_ap.shape[3]
    expected_ap = (*raw_ap.shape[:3], ap_count)
    if loaded["denoised AP"].shape != expected_ap:
        raise QCError("denoised AP measurement count must match raw AP")
    if loaded["Gibbs AP"].shape != expected_ap:
        raise QCError("Gibbs AP measurement count must match raw AP")
    if loaded["EDDY DWI"].shape != raw_pa.shape:
        raise QCError("EDDY DWI shape must match raw PA")

    brain = _image_array(loaded["brain mask"], "brain mask")
    if not np.equal(brain, np.rint(brain)).all() or not np.isin(brain, (0, 1)).all():
        raise QCError("brain mask must be finite, integral, binary 0/1")
    if not np.any(brain == 1):
        raise QCError("brain mask must be nonempty")

    atlas = _image_array(loaded["warped atlas"], "warped atlas")
    if not np.equal(atlas, np.rint(atlas)).all():
        raise QCError("warped atlas must be exactly integer-valued")
    labels = tuple(int(value) for value in np.unique(atlas) if value != 0)
    if labels != tuple(range(1, 49)):
        raise QCError("warped atlas must have nonzero labels exactly 1 through 48")

    topup_manifest = _load_json(
        frozen(context.topup_manifest_json), "TOPUP manifest"
    )
    pa_count, ap_count_manifest = _validate_topup_manifest(
        topup_manifest,
        loaded["TOPUP merged b0"].shape[3],
        loaded["TOPUP corrected b0"].shape[3],
    )
    if pa_count != int(np.count_nonzero(bvals < 50.0)):
        raise QCError("TOPUP manifest PA count must equal raw PA b0 count")
    if ap_count_manifest != ap_count:
        raise QCError("TOPUP manifest AP count must equal raw AP count")

    stripe_rows = _load_stripe_csv(
        frozen(context.stripe_metrics_csv), bvals, int(raw_pa.shape[0])
    )
    stripe_decision = _load_json(
        frozen(context.stripe_decision_json), "stripe decision"
    )
    _validate_stripe_decision(stripe_decision, context, stripe_rows)

    parameters = _load_numeric_table(
        frozen(context.eddy_parameters), "EDDY parameters"
    )
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    if parameters.shape[0] != raw_pa.shape[3] or parameters.shape[1] < 6:
        raise QCError("EDDY parameters must have one row per DWI and at least six columns")
    movement = _load_numeric_table(
        frozen(context.eddy_movement_rms), "EDDY movement RMS"
    )
    if movement.ndim == 1:
        movement = movement[None, :]
    if movement.shape[0] != raw_pa.shape[3] or movement.shape[1] < 2:
        raise QCError("EDDY movement RMS must have one row per DWI and two columns")
    if np.any(movement[:, :2] < 0):
        raise QCError("EDDY movement RMS values must be nonnegative")
    outliers = _load_outlier_map(
        frozen(context.eddy_outlier_map), raw_pa.shape[3]
    )

    return _QCInputs(
        bvals=_readonly(bvals),
        raw_pa=raw_pa,
        raw_ap=raw_ap,
        brain=_readonly(brain),
        affine=np.asarray(raw_pa.affine, dtype=np.float64),
        topup_manifest=MappingProxyType(dict(topup_manifest)),
        stripe_rows=stripe_rows,
        stripe_decision=MappingProxyType(dict(stripe_decision)),
        eddy_parameters=_readonly(parameters),
        movement_rms=_readonly(movement),
        outlier_map=_readonly(outliers),
        snapshots=snapshots.paths,
    )


def _validate_topup_manifest(
    payload: Mapping[str, object], merged_count: int, corrected_count: int
) -> tuple[int, int]:
    pa = payload.get("pa_b0_count")
    ap = payload.get("ap_b0_count")
    combined = payload.get("combined_b0_count")
    order = payload.get("volume_order")
    eddy_order = payload.get("eddy_acquisition_row_order")
    if (
        isinstance(pa, bool)
        or not isinstance(pa, int)
        or pa < 1
        or isinstance(ap, bool)
        or not isinstance(ap, int)
        or ap < 1
        or combined != pa + ap
        or merged_count != pa + ap
        or corrected_count != pa + ap
    ):
        raise QCError("TOPUP manifest has invalid PA/AP volume counts")
    if order != ["PA"] * pa + ["AP"] * ap or eddy_order != ["PA", "AP"]:
        raise QCError("TOPUP manifest must explicitly record PA-then-AP order")
    return pa, ap


def _load_stripe_csv(
    path: Path, bvals: np.ndarray, sagittal_size: int
) -> tuple[Mapping[str, object], ...]:
    required = {
        "volume_index_zero_based",
        "volume_number_one_based",
        "b_value",
        "nominal_shell",
        "a_si",
        "c_si",
        "classification",
        "peak_sagittal_index_zero_based",
        "peak_sagittal_number_one_based",
    }
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or set(reader.fieldnames) != required
                or len(reader.fieldnames) != len(required)
            ):
                raise QCError("stripe CSV has missing or unexpected columns")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise QCError("cannot read stripe metrics CSV") from error
    if len(raw_rows) != bvals.size:
        raise QCError("stripe CSV row count must equal DWI volume count")
    rows: list[Mapping[str, object]] = []
    for index, row in enumerate(raw_rows):
        try:
            zero = int(row["volume_index_zero_based"])
            one = int(row["volume_number_one_based"])
            b_value = float(row["b_value"])
            shell = int(row["nominal_shell"])
            a_si = float(row["a_si"])
            c_si = float(row["c_si"])
            peak_zero = int(row["peak_sagittal_index_zero_based"])
            peak_one = int(row["peak_sagittal_number_one_based"])
        except (TypeError, ValueError) as error:
            raise QCError("stripe CSV contains malformed numeric values") from error
        classification = row["classification"]
        numbers = (b_value, a_si, c_si)
        if (
            not np.isfinite(numbers).all()
            or zero != index
            or one != index + 1
            or peak_one != peak_zero + 1
            or peak_zero < 0
            or peak_zero >= sagittal_size
            or a_si < 0
            or c_si <= 0
            or not math.isclose(b_value, float(bvals[index]), abs_tol=1e-6)
            or shell != int(round_shells([bvals[index]])[0])
            or classification != _classify(c_si)
        ):
            raise QCError("stripe CSV values or indices are inconsistent")
        rows.append(
            MappingProxyType(
                {
                    "index": zero,
                    "number": one,
                    "b_value": b_value,
                    "shell": shell,
                    "a_si": a_si,
                    "c_si": c_si,
                    "classification": classification,
                }
            )
        )
    return tuple(rows)


def _validate_stripe_decision(
    payload: Mapping[str, object],
    context: StageQCContext,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if payload.get("subject_id") != context.stage_context.config.subject_id:
        raise QCError("stripe decision subject_id does not match configuration")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or thresholds != {
        "ambiguous_min_inclusive": _AMBIGUOUS,
        "high_min_exclusive": _HIGH,
        "exclude_high_volume_count": 5,
    }:
        raise QCError("stripe decision thresholds do not match the accepted procedure")
    high = [int(row["index"]) for row in rows if row["classification"] == "high"]
    ambiguous = [
        int(row["index"]) for row in rows if row["classification"] == "ambiguous"
    ]
    flagged = payload.get("flagged_indices_zero_based")
    one_based = payload.get("flagged_volume_numbers_one_based")
    counts = payload.get("volume_counts")
    maximum = payload.get("maximum_csi")
    if flagged != {"high": high, "ambiguous": ambiguous}:
        raise QCError("stripe decision flagged indices disagree with CSV")
    if one_based != {
        "high": [value + 1 for value in high],
        "ambiguous": [value + 1 for value in ambiguous],
    }:
        raise QCError("stripe decision one-based volumes disagree with CSV")
    normal_count = len(rows) - len(high) - len(ambiguous)
    if counts != {
        "total": len(rows),
        "normal": normal_count,
        "ambiguous": len(ambiguous),
        "high": len(high),
    }:
        raise QCError("stripe decision volume counts disagree with CSV")
    reviewed = context.stage_context.config.analysis.ambiguous_qc_reviewed
    expected_status = (
        "EXCLUDE"
        if len(high) >= 5
        else "HOLD_FOR_REVIEW"
        if ambiguous and not reviewed
        else "INCLUDE_AFTER_REVIEW"
        if ambiguous
        else "INCLUDE_WITH_FLAGS"
        if high
        else "INCLUDE"
    )
    expected_exit = 20 if expected_status == "EXCLUDE" else 21 if expected_status == "HOLD_FOR_REVIEW" else 0
    if (
        payload.get("ambiguous_reviewed") is not reviewed
        or payload.get("decision") != expected_status
        or payload.get("exit_code") != expected_exit
    ):
        raise QCError("stripe decision gate status is inconsistent with cSI values")
    maximum_index = int(np.argmax([float(row["c_si"]) for row in rows]))
    expected_max = float(rows[maximum_index]["c_si"])
    if (
        not isinstance(maximum, dict)
        or maximum.get("volume_index_zero_based") != maximum_index
        or maximum.get("volume_number_one_based") != maximum_index + 1
        or not math.isclose(float(maximum.get("value", math.nan)), expected_max)
    ):
        raise QCError("stripe decision maximum cSI disagrees with CSV")


def _load_outlier_map(path: Path, volumes: int) -> np.ndarray:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise QCError("cannot read EDDY outlier map") from error
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        raise QCError("EDDY outlier map is empty")
    first_tokens = nonempty[0].split()
    header = any(_not_number(token) for token in first_tokens)
    data_lines = nonempty[1:] if header else nonempty
    try:
        values = np.asarray(
            [[float(token) for token in line.split()] for line in data_lines],
            dtype=np.float64,
        )
    except (ValueError, TypeError) as error:
        raise QCError("EDDY outlier map contains malformed values") from error
    if values.ndim != 2 or values.shape[0] != volumes or values.shape[1] < 1:
        raise QCError("EDDY outlier map must have one row per DWI volume")
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.rint(values)).all()
        or not np.isin(values, (0, 1)).all()
    ):
        raise QCError("EDDY outlier map must contain finite integral 0/1 values")
    return values


def _render_figures(
    context: StageQCContext,
    inputs: _QCInputs,
    figure_ids: Sequence[str],
    temporary: _OwnedDirectory,
) -> list[tuple[Path, str]]:
    renderers: Mapping[
        str,
        Callable[
            [StageQCContext, _QCInputs, Path | _OwnedWriteHandle],
            None,
        ],
    ] = {
        "input": _render_input,
        "shell_distribution": _render_shells,
        "stripe": _render_stripe,
        "denoise_pa": lambda c, i, p: _render_comparison(
            _frozen(i, c.raw_pa),
            _frozen(i, c.denoised_pa),
            i.bvals,
            i.brain,
            "PA MP-PCA",
            p,
        ),
        "denoise_ap": lambda c, i, p: _render_comparison(
            _frozen(i, c.raw_ap),
            _frozen(i, c.denoised_ap),
            None,
            i.brain,
            "AP MP-PCA",
            p,
        ),
        "gibbs_pa": lambda c, i, p: _render_comparison(
            _frozen(i, c.denoised_pa),
            _frozen(i, c.gibbs_pa),
            i.bvals,
            i.brain,
            "PA Gibbs correction",
            p,
        ),
        "gibbs_ap": lambda c, i, p: _render_comparison(
            _frozen(i, c.denoised_ap),
            _frozen(i, c.gibbs_ap),
            None,
            i.brain,
            "AP Gibbs correction",
            p,
        ),
        "topup": _render_topup,
        "bet": _render_bet,
        "eddy_images": _render_eddy_images,
        "eddy_motion": _render_eddy_motion,
        "dti": _render_dti,
        "dki": _render_dki,
        "dki_direct": _render_dki_direct,
        "noddi": _render_noddi,
        "jhu_48roi": _render_atlas,
    }
    rendered: list[tuple[Path, str]] = []
    for figure_id in figure_ids:
        if figure_id not in renderers:
            raise QCError(f"figure cannot be rendered directly: {figure_id}")
        basename = _figure_basename(context, figure_id)
        target = temporary.path / basename
        temporary.verify()
        with temporary.writer(basename) as output:
            renderers[figure_id](context, inputs, output)
        temporary.verify()
        _validate_owned_png(temporary, basename)
        rendered.append((target, basename))
    return rendered


def _frozen(inputs: _QCInputs, original: Path) -> Path:
    try:
        return inputs.snapshots[Path(original)]
    except KeyError as error:
        raise QCError("render attempted to read an undeclared live input") from error


def _render_input(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    pa = _mean_b0(inputs.raw_pa, inputs.bvals)
    ap = _mean_image(inputs.raw_ap)
    pa, ap, mask = _canonical_triplet(pa, ap, inputs.raw_pa.affine, inputs.brain)
    location = _slice_location(mask, pa)
    values = np.concatenate((pa[np.isfinite(pa)], ap[np.isfinite(ap)]))
    vmin, vmax = _percentiles(values)
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), constrained_layout=True)
    for row, (data, label) in enumerate(((pa, "PA mean b0"), (ap, "AP mean b0"))):
        for col, (view, plane) in enumerate(_three_views(data, location)):
            axes[row, col].imshow(view, cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{label} — {plane}")
            axes[row, col].axis("off")
    figure.suptitle("Input b0 phase-encoding pair")
    _save_figure(figure, path)


def _render_shells(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    shells = round_shells(inputs.bvals)
    values, counts = np.unique(shells, return_counts=True)
    figure, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    bars = axis.bar(np.arange(values.size), counts, color="0.35")
    axis.set_xticks(
        np.arange(values.size),
        ["b0" if value == 0 else str(int(value)) for value in values],
    )
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            float(bar.get_height()),
            str(int(count)),
            ha="center",
            va="bottom",
        )
    axis.set_xlabel("Nominal shell (s/mm²)")
    axis.set_ylabel("DWI volume count")
    axis.set_title("Acquired shell distribution")
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def _render_stripe(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    rows = inputs.stripe_rows
    numbers = np.asarray([row["number"] for row in rows], dtype=float)
    csi = np.asarray([row["c_si"] for row in rows], dtype=float)
    shells = np.asarray([row["shell"] for row in rows], dtype=int)
    classes = [str(row["classification"]) for row in rows]
    styles = {
        "normal": ("o", "0.25"),
        "ambiguous": ("^", "darkorange"),
        "high": ("X", "crimson"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for name in ("normal", "ambiguous", "high"):
        indices = np.asarray([value == name for value in classes])
        marker, color = styles[name]
        axes[0].scatter(
            numbers[indices], csi[indices], marker=marker, color=color, label=name
        )
        if name != "normal":
            for number, value in zip(numbers[indices], csi[indices], strict=True):
                axes[0].annotate(
                    str(int(number)), (number, value), xytext=(3, 3),
                    textcoords="offset points", fontsize=8
                )
    unique_shells = np.unique(shells)
    for position, shell in enumerate(unique_shells):
        shell_values = csi[shells == shell]
        axes[1].scatter(
            np.full(shell_values.size, position, dtype=float),
            shell_values,
            marker="o",
            facecolors="none",
            edgecolors="0.25",
        )
    axes[1].set_xticks(
        np.arange(unique_shells.size),
        ["b0" if value == 0 else str(int(value)) for value in unique_shells],
    )
    for axis in axes:
        axis.axhline(_AMBIGUOUS, color="darkorange", linestyle="--", label="1.15")
        axis.axhline(_HIGH, color="crimson", linestyle=":", label="1.25")
        axis.set_ylabel("Corrected stripe index (cSI)")
        axis.grid(axis="y", color="0.88", linewidth=0.6)
    axes[0].set_xlabel("DWI volume number (one-based)")
    axes[0].set_title("cSI by volume")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Nominal shell (s/mm²)")
    axes[1].set_title("cSI by shell")
    decision = str(inputs.stripe_decision.get("decision"))
    high = [int(row["number"]) for row in rows if row["classification"] == "high"]
    ambiguous = [
        int(row["number"]) for row in rows if row["classification"] == "ambiguous"
    ]
    figure.suptitle(
        f"Pre-denoise motion/stripe screen — {decision}; "
        f"high {high or 'none'}, ambiguous {ambiguous or 'none'}"
    )
    _save_figure(figure, path)


def _render_comparison(
    before_path: Path,
    after_path: Path,
    bvals: np.ndarray | None,
    brain: np.ndarray,
    title: str,
    path: Path,
) -> None:
    before_image = _load_nifti(before_path, "comparison before", dimensions=(3, 4))
    after_image = _load_nifti(after_path, "comparison after", dimensions=(3, 4))
    before = _mean_b0(before_image, bvals) if bvals is not None else _mean_image(before_image)
    after = _mean_b0(after_image, bvals) if bvals is not None else _mean_image(after_image)
    before, after, mask = _canonical_triplet(
        before, after, before_image.affine, brain
    )
    location = _slice_location(mask, before)
    values = np.concatenate((before[mask > 0], after[mask > 0]))
    vmin, vmax = _percentiles(values)
    difference = after - before
    dmax = _symmetric_limit(difference[mask > 0])
    figure, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    rows = (
        (before, "Before", "gray", vmin, vmax),
        (after, "After", "gray", vmin, vmax),
        (difference, "After − before", "coolwarm", -dmax, dmax),
    )
    for row, (data, label, cmap, lower, upper) in enumerate(rows):
        for col, (view, plane) in enumerate(_three_views(data, location)):
            axes[row, col].imshow(view, cmap=cmap, vmin=lower, vmax=upper)
            axes[row, col].set_title(f"{label} — {plane}")
            axes[row, col].axis("off")
    figure.suptitle(title)
    _save_figure(figure, path)


def _render_topup(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    merged_image = _load_nifti(
        _frozen(inputs, context.topup_merged_b0), "merged b0", dimensions=(4,)
    )
    corrected_image = _load_nifti(
        _frozen(inputs, context.topup_corrected_b0),
        "corrected b0",
        dimensions=(4,),
    )
    pa_count = int(inputs.topup_manifest["pa_b0_count"])
    ap_count = int(inputs.topup_manifest["ap_b0_count"])
    before_pa = _mean_range(merged_image, range(pa_count))
    before_ap = _mean_range(merged_image, range(pa_count, pa_count + ap_count))
    after_pa = _mean_range(corrected_image, range(pa_count))
    after_ap = _mean_range(corrected_image, range(pa_count, pa_count + ap_count))
    arrays = [
        _canonical_3d(value, merged_image.affine)
        for value in (before_pa, after_pa, before_ap, after_ap)
    ]
    mask = _canonical_3d(inputs.brain, inputs.raw_pa.affine)
    location = _slice_location(mask, arrays[0])
    vmin, vmax = _percentiles(np.concatenate([value[mask > 0] for value in arrays]))
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.2), constrained_layout=True)
    columns = (
        (arrays[0], "PA before"),
        (arrays[1], "PA after"),
        (arrays[2], "AP before"),
        (arrays[3], "AP after"),
    )
    for column, (data, label) in enumerate(columns):
        views = {plane: view for view, plane in _three_views(data, location)}
        for row, plane in enumerate(("Coronal", "Axial")):
            axes[row, column].imshow(views[plane], cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, column].set_title(f"{label} — {plane}")
            axes[row, column].axis("off")
    figure.suptitle(f"TOPUP correction — {pa_count} PA and {ap_count} AP b0 volumes")
    _save_figure(figure, path)


def _render_bet(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    image = _load_nifti(
        _frozen(inputs, context.hifi_b0), "hifi b0", dimensions=(3,)
    )
    data = _canonical_3d(_image_array(image, "hifi b0"), image.affine)
    mask = _canonical_3d(inputs.brain, inputs.raw_pa.affine)
    location = _slice_location(mask, data)
    vmin, vmax = _percentiles(data[mask > 0])
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    for axis, (view, plane), (mask_view, _) in zip(
        axes, _three_views(data, location), _three_views(mask, location), strict=True
    ):
        axis.imshow(view, cmap="gray", vmin=vmin, vmax=vmax)
        boundary = _binary_boundary(mask_view > 0)
        overlay = np.zeros((*boundary.shape, 4), dtype=np.float32)
        overlay[boundary] = (1.0, 1.0, 0.0, 1.0)
        axis.imshow(overlay, interpolation="nearest")
        axis.set_title(f"{plane} — mask boundary")
        axis.axis("off")
    figure.suptitle("BET cleaned brain mask overlay")
    _save_figure(figure, path)


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Return an in-mask boundary, treating the image border as background."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise QCError("binary boundary input must be two-dimensional")
    eroded = ndimage.binary_erosion(
        binary,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    return binary & ~eroded


def _render_eddy_images(
    context: StageQCContext, inputs: _QCInputs, path: Path
) -> None:
    _render_comparison(
        _frozen(inputs, context.gibbs_pa),
        _frozen(inputs, context.eddy_dwi),
        inputs.bvals,
        inputs.brain,
        "EDDY mean b0 correction",
        path,
    )


def _render_eddy_motion(
    context: StageQCContext, inputs: _QCInputs, path: Path
) -> None:
    parameters = inputs.eddy_parameters
    translation = parameters[:, :3]
    rotation = np.rad2deg(parameters[:, 3:6])
    rms = inputs.movement_rms[:, :2]
    outlier_counts = np.sum(inputs.outlier_map, axis=1)
    volumes = np.arange(1, parameters.shape[0] + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(volumes, translation)
    axes[0, 0].set_title(
        f"Translation (mm), max |value| {np.max(np.abs(translation)):.3g}"
    )
    axes[0, 0].legend(("x", "y", "z"), fontsize=8)
    axes[0, 1].plot(volumes, rotation)
    axes[0, 1].set_title(
        f"Rotation (degrees), max |value| {np.max(np.abs(rotation)):.3g}"
    )
    axes[0, 1].legend(("x", "y", "z"), fontsize=8)
    axes[1, 0].plot(volumes, rms)
    axes[1, 0].set_title(
        f"RMS (mm), maxima abs {np.max(rms[:, 0]):.3g}, rel {np.max(rms[:, 1]):.3g}"
    )
    axes[1, 0].legend(("absolute", "relative"), fontsize=8)
    axes[1, 1].bar(volumes, outlier_counts, color="0.35")
    affected = int(np.count_nonzero(outlier_counts))
    axes[1, 1].set_title(
        f"Outlier slices — {affected} affected volumes; max {int(outlier_counts.max())}"
    )
    for axis in axes.ravel():
        axis.set_xlabel("DWI volume number (one-based)")
        axis.grid(axis="y", color="0.88", linewidth=0.6)
    figure.suptitle("EDDY motion and slice-outlier technical QC")
    _save_figure(figure, path)


def _render_dti(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    _render_metric_family(
        {key: _frozen(inputs, value) for key, value in context.dti_maps.items()},
        DTI_KEYS,
        inputs,
        "DTI parameter maps",
        path,
        bounded={"FA"},
    )


def _render_dki(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    _render_metric_family(
        {key: _frozen(inputs, value) for key, value in context.dki_maps.items()},
        DKI_KEYS,
        inputs,
        "DKI parameter maps",
        path,
        bounded={"FA"},
    )


def _render_dki_direct(
    context: StageQCContext, inputs: _QCInputs, path: Path
) -> None:
    _render_metric_family(
        {
            key: _frozen(inputs, value)
            for key, value in context.dki_direct_maps.items()
        },
        DKI_DIRECT_KEYS,
        inputs,
        "Direct average-signal DKI parameter maps",
        path,
        bounded=set(),
    )


def _render_noddi(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    _render_metric_family(
        {key: _frozen(inputs, value) for key, value in context.noddi_maps.items()},
        NODDI_KEYS,
        inputs,
        "NODDI parameter maps",
        path,
        bounded=set(NODDI_KEYS),
    )


def _render_metric_family(
    maps: Mapping[str, Path],
    keys: Sequence[str],
    inputs: _QCInputs,
    title: str,
    path: Path,
    *,
    bounded: set[str],
) -> None:
    mask = _canonical_3d(inputs.brain, inputs.raw_pa.affine)
    location = _slice_location(mask, mask)
    columns = min(4, len(keys))
    rows = int(math.ceil(len(keys) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 3.1 * rows), constrained_layout=True
    )
    flat = np.atleast_1d(axes).ravel()
    for axis, key in zip(flat, keys, strict=False):
        image = _load_nifti(maps[key], key, dimensions=(3,))
        data = _canonical_3d(_image_array(image, key), image.affine)
        values = data[mask > 0]
        if key in bounded:
            lower, upper = 0.0, 1.0
        else:
            lower, upper = _percentiles(values)
        axial = {plane: view for view, plane in _three_views(data, location)}["Axial"]
        shown = axis.imshow(axial, cmap="viridis", vmin=lower, vmax=upper)
        unit = "mm²/s" if key in {"MD", "AD", "RD"} else "a.u."
        axis.set_title(f"{key} ({unit})")
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.045, pad=0.02)
    for axis in flat[len(keys) :]:
        axis.axis("off")
    figure.suptitle(title)
    _save_figure(figure, path)


def _render_atlas(context: StageQCContext, inputs: _QCInputs, path: Path) -> None:
    fa_image = _load_nifti(
        _frozen(inputs, context.dti_maps["FA"]), "DTI FA", dimensions=(3,)
    )
    atlas_image = _load_nifti(
        _frozen(inputs, context.warped_atlas), "JHU atlas", dimensions=(3,)
    )
    fa = _canonical_3d(_image_array(fa_image, "DTI FA"), fa_image.affine)
    atlas = _canonical_3d(
        _image_array(atlas_image, "JHU atlas"), atlas_image.affine, order=0
    )
    mask = _canonical_3d(inputs.brain, inputs.raw_pa.affine)
    location = _slice_location(atlas > 0, fa)
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    cmap = plt.get_cmap("turbo", 48)
    for axis, (fa_view, plane), (atlas_view, _) in zip(
        axes, _three_views(fa, location), _three_views(atlas, location), strict=True
    ):
        axis.imshow(fa_view, cmap="gray", vmin=0, vmax=1)
        overlay = np.ma.masked_where(atlas_view == 0, atlas_view)
        axis.imshow(overlay, cmap=cmap, vmin=1, vmax=48, alpha=0.48, interpolation="nearest")
        axis.set_title(f"{plane} — discrete labels 1–48")
        axis.axis("off")
    figure.suptitle("JHU 48-ROI nearest-neighbour alignment overlay")
    _save_figure(figure, path)


def _render_overview(
    paths: Sequence[Path],
    output: Path | _OwnedWriteHandle,
    *,
    source_owner: _OwnedDirectory | None = None,
) -> None:
    if len(paths) != 16:
        raise QCError("overview requires exactly 16 non-overview figures")
    figure, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    for axis, figure_id, path in zip(
        axes.ravel(), NON_OVERVIEW_FIGURE_IDS, paths, strict=True
    ):
        if source_owner is None:
            _validate_png(path)
            pixels = mpimg.imread(path)
        else:
            source_owner.verify()
            pixels = _decode_png_bytes(
                source_owner.read_bytes(path.name),
                path.name,
            )
        axis.imshow(pixels)
        axis.set_title(figure_id.replace("_", " "), fontsize=9)
        axis.axis("off")
    figure.suptitle("Complete stepwise diffusion MRI technical QC overview")
    _save_figure(figure, output, dpi=120)


def _write_qc_manifest(
    context: StageQCContext,
    rendered: Sequence[tuple[Path, str]],
    owner: _OwnedDirectory,
    name: str,
) -> None:
    if len(rendered) != len(FIGURE_IDS):
        raise QCError("QC manifest requires exactly 17 figures")
    figures: OrderedDict[str, str] = OrderedDict()
    metadata: OrderedDict[str, dict[str, object]] = OrderedDict()
    for figure_id, (source, basename) in zip(FIGURE_IDS, rendered, strict=True):
        expected = _figure_basename(context, figure_id)
        if basename != expected:
            raise QCError("QC figure basename does not match the frozen manifest")
        width, height = _validate_owned_png(owner, basename)
        figures[figure_id] = basename
        metadata[figure_id] = {
            "sha256": _hash_descriptor(owner.descriptor(basename)),
            "width": width,
            "height": height,
        }
    detail_name = context.stripe_metrics_csv.parent.name
    if not detail_name or detail_name in {".", ".."} or "/" in detail_name:
        raise QCError("pre-denoise detail directory has an unsafe name")
    payload = OrderedDict(
        (
            ("schema_version", "1.0"),
            ("subject_id", context.stage_context.config.subject_id),
            ("visual_review_status", "NOT_REVIEWED"),
            ("figures", figures),
            ("figure_metadata", metadata),
            ("pre_denoise_detail_directory", detail_name),
        )
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _reject_serialized_path_leaks(text)
    owner.write_text(name, text)


def _require_safe_basename(name: str, label: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or os.path.basename(name) != name
        or "\x00" in name
    ):
        raise QCError(f"{label} name must be a safe basename")


def _remove_exact_empty_directory(
    parent_descriptor: int,
    identity: tuple[int, int],
    *,
    root_descriptor: int | None = None,
) -> bool:
    """Remove only the empty directory with ``identity`` below a pinned parent."""
    candidates: list[str] = []
    for name in os.listdir(parent_descriptor):
        try:
            details = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            continue
        if (
            stat.S_ISDIR(details.st_mode)
            and (int(details.st_dev), int(details.st_ino)) == identity
        ):
            candidates.append(name)
    if not candidates:
        if root_descriptor is not None:
            details = os.fstat(root_descriptor)
            if (int(details.st_dev), int(details.st_ino)) != identity:
                raise QCError("private directory cleanup root identity changed")
            if details.st_nlink == 0:
                return True
        return False
    if len(candidates) != 1:
        raise QCError("private directory cleanup root identity is ambiguous")

    name = candidates[0]
    if root_descriptor is None:
        named = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named.st_mode)
            or (int(named.st_dev), int(named.st_ino)) != identity
        ):
            raise QCError("private directory cleanup root binding changed")
        os.rmdir(name, dir_fd=parent_descriptor)
        return True

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    candidate_descriptor = os.open(
        name,
        flags,
        dir_fd=parent_descriptor,
    )
    try:
        details = os.fstat(candidate_descriptor)
        if (
            not stat.S_ISDIR(details.st_mode)
            or (int(details.st_dev), int(details.st_ino)) != identity
            or (
                root_descriptor is not None
                and _fd_identity(root_descriptor) != identity
            )
        ):
            raise QCError("private directory cleanup root identity changed")
        if os.listdir(candidate_descriptor):
            return False
        named = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named.st_mode)
            or (int(named.st_dev), int(named.st_ino)) != identity
        ):
            raise QCError("private directory cleanup root binding changed")
        os.rmdir(name, dir_fd=parent_descriptor)
        return True
    finally:
        os.close(candidate_descriptor)


def _require_empty_destination(path: Path) -> _DirectoryAnchor:
    return _pin_output_directory(path, require_empty=True)


def _require_overview_prerequisites(
    context: StageQCContext,
) -> _DirectoryAnchor:
    anchor = _pin_output_directory(
        context.output_directory,
        require_empty=False,
    )
    try:
        expected = {
            _figure_basename(context, value)
            for value in NON_OVERVIEW_FIGURE_IDS
        }
        actual = set(os.listdir(anchor.root_descriptor))
        if actual != expected:
            raise QCError(
                "overview requires exactly all 16 non-overview QC figures"
            )
        for path in _non_overview_paths(context):
            _require_regular(path)
        anchor.verify_nominal_binding()
        if _figure_path(context, "overview").exists():
            raise QCError("overview output already exists")
        return anchor
    except BaseException:
        anchor.close()
        raise


def _pin_output_directory(
    path: Path,
    *,
    require_empty: bool,
    label: str = "QC destination",
) -> _DirectoryAnchor:
    _reject_traversal(path)
    _reject_symlink_components(path)
    try:
        anchor = _DirectoryAnchor.bind(
            path,
            label=label,
        )
    except OSError as error:
        raise QCError(f"{label} must already exist and be pinnable") from error
    try:
        os.fchmod(anchor.root_descriptor, 0o700)
        anchor.verify_nominal_binding()
        if require_empty and os.listdir(anchor.root_descriptor):
            raise QCError(
                "QC destination must be empty and outputs are never overwritten"
            )
        return anchor
    except BaseException:
        anchor.close()
        raise


def _private_temp(
    parent: Path,
    *,
    names: Sequence[str] = (),
) -> _OwnedDirectory:
    try:
        return _OwnedDirectory.create(
            parent,
            prefix=".dmri-qc-",
            names=names,
        )
    except QCError:
        raise
    except OSError as error:
        raise QCError(
            "cannot create private QC temporary directory"
        ) from error


def _commit_files(
    destination: Path | _DirectoryAnchor,
    sources: Sequence[tuple[Path, str]],
    *,
    require_empty: bool = False,
    verifier: Callable[[], None] | None = None,
    source_owner: _OwnedDirectory | None = None,
) -> None:
    names = [name for _, name in sources]
    if len(set(names)) != len(names):
        raise QCError("QC output names must be unique safe basenames")
    for name in names:
        _require_safe_basename(name, "QC output")
    owns_anchor = not isinstance(destination, _DirectoryAnchor)
    anchor = (
        _pin_output_directory(
            Path(destination),
            require_empty=require_empty,
        )
        if owns_anchor
        else destination
    )
    created: list[_PendingCommitFile] = []
    source_digests: dict[str, str] = {}
    try:
        anchor.verify_nominal_binding()
        if verifier is not None:
            verifier()
        if source_owner is not None:
            source_owner.verify()
        if require_empty and os.listdir(anchor.root_descriptor):
            raise QCError("QC destination became nonempty before commit")
        for source, name in sources:
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=anchor.root_descriptor,
            )
            owned = _PendingCommitFile(name, descriptor)
            created.append(owned)
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise QCError("created QC output is not a private regular file")
            owned.identity = (
                int(details.st_dev),
                int(details.st_ino),
            )
            source_digests[name] = _copy_commit_source(
                source,
                descriptor,
                source_owner=source_owner,
            )
            os.fsync(descriptor)
        os.fsync(anchor.root_descriptor)
        anchor.verify_nominal_binding()
        _verify_committed_outputs(
            anchor,
            created,
            sources,
            source_digests,
        )
        if source_owner is not None:
            source_owner.verify()
        if verifier is not None:
            verifier()
        anchor.verify_nominal_binding()
        _verify_committed_outputs(
            anchor,
            created,
            sources,
            source_digests,
        )
    except BaseException as primary:
        try:
            cleanup_error = _rollback_created_outputs(
                anchor.root_descriptor,
                created,
            )
        except BaseException as unexpected_cleanup:
            cleanup_error = QCError(
                "owned QC output rollback raised unexpectedly"
            )
            cleanup_error.__cause__ = unexpected_cleanup
        if isinstance(primary, OSError):
            if primary.errno == errno.EEXIST:
                mapped = QCError(
                    "QC output already exists; refusing to overwrite"
                )
            else:
                mapped = QCError(
                    "failed to commit QC outputs transactionally"
                )
            if cleanup_error is not None and hasattr(mapped, "add_note"):
                mapped.add_note(f"rollback cleanup failed: {cleanup_error}")
            raise mapped from primary
        if cleanup_error is not None and hasattr(primary, "add_note"):
            primary.add_note(f"rollback cleanup failed: {cleanup_error}")
        raise
    finally:
        for owned in created:
            try:
                os.close(owned.descriptor)
            except OSError:
                pass
        if owns_anchor:
            anchor.close()


def _copy_commit_source(
    source: Path,
    output_descriptor: int,
    *,
    source_owner: _OwnedDirectory | None,
) -> str:
    close_source = source_owner is None
    if source_owner is None:
        try:
            source_descriptor = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as error:
            raise QCError(
                f"cannot open staged QC output: {source.name}"
            ) from error
    else:
        source_owner.verify()
        source_descriptor = source_owner.descriptor(source.name)
    try:
        details = os.fstat(source_descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise QCError("staged QC output is not a regular file")
        digest = _hash_descriptor(source_descriptor)
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while chunk := os.read(source_descriptor, 1024 * 1024):
            _write_all(output_descriptor, chunk)
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        if _hash_descriptor(source_descriptor) != digest:
            raise QCError("staged QC output changed during commit")
        return digest
    finally:
        if close_source:
            os.close(source_descriptor)


def _verify_committed_outputs(
    anchor: _DirectoryAnchor,
    created: Sequence[_PendingCommitFile],
    sources: Sequence[tuple[Path, str]],
    source_digests: Mapping[str, str],
) -> None:
    created_by_name = {owned.name: owned for owned in created}
    if set(created_by_name) != {name for _, name in sources}:
        raise QCError("committed QC output set is incomplete")
    for source, name in sources:
        owned = created_by_name[name]
        if owned.identity is None:
            raise QCError("committed QC output identity is unavailable")
        descriptor_details = os.fstat(owned.descriptor)
        details = os.stat(
            name,
            dir_fd=anchor.root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(descriptor_details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or descriptor_details.st_nlink != 1
            or details.st_nlink != 1
            or _fd_identity(owned.descriptor) != owned.identity
            or (int(details.st_dev), int(details.st_ino))
            != owned.identity
        ):
            raise QCError("committed QC output identity changed")
        if _hash_descriptor(owned.descriptor) != source_digests[name]:
            raise QCError("committed QC output digest mismatch")
        if name.endswith(".png"):
            anchor.verify_nominal_binding()
            _validate_png(anchor.path / name)
            anchor.verify_nominal_binding()
            checked = os.stat(
                name,
                dir_fd=anchor.root_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(checked.st_mode)
                or checked.st_nlink != 1
                or (int(checked.st_dev), int(checked.st_ino))
                != owned.identity
            ):
                raise QCError("committed QC output identity changed")


def _rollback_created_outputs(
    directory_descriptor: int,
    created: Sequence[_PendingCommitFile],
) -> QCError | None:
    if not created:
        return None
    unresolved: list[int] = []
    for owned in created:
        if owned.identity is not None:
            continue
        try:
            details = _recover_descriptor_stat(owned.descriptor)
            if not stat.S_ISREG(details.st_mode):
                unresolved.append(owned.descriptor)
                continue
            owned.identity = (
                int(details.st_dev),
                int(details.st_ino),
            )
        except BaseException:
            unresolved.append(owned.descriptor)
    identities = {
        owned.identity
        for owned in created
        if owned.identity is not None
    }
    for owned in created:
        try:
            details = os.fstat(owned.descriptor)
            if (
                owned.identity is not None
                and stat.S_ISREG(details.st_mode)
                and _fd_identity(owned.descriptor) == owned.identity
            ):
                os.ftruncate(owned.descriptor, 0)
                os.fsync(owned.descriptor)
        except OSError:
            pass

    final_errors: list[BaseException] = []
    remaining_identities = set(identities)
    for _attempt in range(3):
        final_errors = []
        try:
            names = os.listdir(directory_descriptor)
        except OSError as error:
            final_errors.append(error)
            continue
        for name in names:
            try:
                details = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                identity = (int(details.st_dev), int(details.st_ino))
                if (
                    not stat.S_ISREG(details.st_mode)
                    or identity not in identities
                ):
                    continue
                checked = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(checked.st_mode)
                    and (int(checked.st_dev), int(checked.st_ino))
                    == identity
                ):
                    os.unlink(name, dir_fd=directory_descriptor)
            except OSError as error:
                final_errors.append(error)

        remaining_identities = set()
        try:
            remaining_names = os.listdir(directory_descriptor)
        except OSError as error:
            final_errors.append(error)
            remaining_names = []
            remaining_identities = set(identities)
        for name in remaining_names:
            try:
                details = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                final_errors.append(error)
                continue
            identity = (int(details.st_dev), int(details.st_ino))
            if stat.S_ISREG(details.st_mode) and identity in identities:
                remaining_identities.add(identity)
        if not remaining_identities and not final_errors:
            break

    unsafe_descriptors: list[int] = []
    for owned in created:
        if owned.identity is None:
            unsafe_descriptors.append(owned.descriptor)
            continue
        try:
            details = os.fstat(owned.descriptor)
        except BaseException as error:
            final_errors.append(error)
            continue
        if (
            _fd_identity(owned.descriptor) != owned.identity
            or details.st_nlink > 0
            or details.st_size > 0
        ):
            unsafe_descriptors.append(owned.descriptor)
    if (
        unresolved
        or remaining_identities
        or unsafe_descriptors
        or final_errors
    ):
        return QCError(
            "owned QC output rollback is incomplete; retry is required"
        )
    return None


def _verify_directory_binding(
    destination: Path, expected: tuple[int, int]
) -> None:
    try:
        descriptor = os.open(
            destination,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise QCError("QC destination was replaced during commit") from error
    try:
        if _fd_identity(descriptor) != expected:
            raise QCError("QC destination was replaced during commit")
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write while committing QC output")
        view = view[written:]


def _save_figure(
    figure: plt.Figure,
    path: Path | _OwnedWriteHandle,
    *,
    dpi: int = 150,
) -> None:
    try:
        figure.savefig(
            path,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "dmri-reproducible-pipeline"},
        )
    except Exception as error:
        raise QCError(f"cannot render QC figure {path.name}") from error
    finally:
        plt.close(figure)


def _validate_png(path: Path) -> tuple[int, int]:
    _require_regular(path)
    try:
        content = path.read_bytes()
    except QCError:
        raise
    except OSError as error:
        raise QCError(f"cannot decode PNG: {path.name}") from error
    return _validate_png_bytes(content, path.name)


def _validate_owned_png(
    owner: _OwnedDirectory,
    name: str,
) -> tuple[int, int]:
    owner.verify()
    return _validate_png_bytes(owner.read_bytes(name), name)


def _decode_png_bytes(content: bytes, name: str) -> np.ndarray:
    if not content.startswith(_PNG_SIGNATURE):
        raise QCError(f"invalid PNG signature: {name}")
    try:
        pixels = np.asarray(
            mpimg.imread(io.BytesIO(content), format="png"),
            dtype=np.float64,
        )
    except (OSError, ValueError, SyntaxError) as error:
        raise QCError(f"cannot decode PNG: {name}") from error
    if pixels.ndim not in (2, 3) or min(pixels.shape[:2]) <= 0:
        raise QCError(f"PNG has zero or invalid dimensions: {name}")
    if not np.isfinite(pixels).all():
        raise QCError(f"PNG contains non-finite pixels: {name}")
    return pixels


def _validate_png_bytes(content: bytes, name: str) -> tuple[int, int]:
    pixels = _decode_png_bytes(content, name)
    return int(pixels.shape[1]), int(pixels.shape[0])


def _load_nifti(
    path: Path, label: str, *, dimensions: tuple[int, ...]
) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(path, mmap=True)
    except (
        OSError,
        ValueError,
        ImageFileError,
        HeaderDataError,
        ImageDataError,
    ) as error:
        raise QCError(f"cannot read {label} NIfTI") from error
    if len(image.shape) not in dimensions or any(int(value) <= 0 for value in image.shape):
        raise QCError(f"{label} must be a nonempty {dimensions}-D NIfTI")
    if not np.isfinite(image.affine).all():
        raise QCError(f"{label} affine must be finite")
    if not np.issubdtype(image.get_data_dtype(), np.number):
        raise QCError(f"{label} must contain numeric data")
    return image


def _require_finite_image(image: nib.spatialimages.SpatialImage, label: str) -> None:
    if len(image.shape) == 4:
        for index in range(image.shape[3]):
            try:
                volume = np.asarray(image.dataobj[..., index], dtype=np.float64)
            except (OSError, ValueError, TypeError, IndexError) as error:
                raise QCError(f"cannot read {label} volume {index + 1}") from error
            if not np.isfinite(volume).all():
                raise QCError(f"{label} contains non-finite image data")
    elif not np.isfinite(_image_array(image, label)).all():
        raise QCError(f"{label} contains non-finite image data")


def _image_array(image: nib.spatialimages.SpatialImage, label: str) -> np.ndarray:
    try:
        return np.asarray(image.dataobj, dtype=np.float64)
    except (OSError, ValueError, TypeError) as error:
        raise QCError(f"cannot read {label} image data") from error


def _require_same_spatial_grid(
    reference: nib.spatialimages.SpatialImage,
    image: nib.spatialimages.SpatialImage,
    label: str,
) -> None:
    if image.shape[:3] != reference.shape[:3] or not np.allclose(
        image.affine, reference.affine, atol=_GRID_ATOL, rtol=0.0
    ):
        raise QCError(f"{label} must use the diffusion spatial grid and affine")


def _mean_b0(
    image: nib.spatialimages.SpatialImage, bvals: np.ndarray | None
) -> np.ndarray:
    if len(image.shape) == 3:
        return _image_array(image, "3D b0")
    if bvals is None or bvals.size != image.shape[3]:
        raise QCError("b-values are required for streaming DWI b0 averaging")
    indices = np.flatnonzero(bvals < 50.0)
    if indices.size == 0:
        raise QCError("DWI has no b0 with b < 50")
    return _mean_range(image, indices)


def _mean_image(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    if len(image.shape) == 3:
        return _image_array(image, "3D image")
    return _mean_range(image, range(image.shape[3]))


def _mean_range(
    image: nib.spatialimages.SpatialImage, indices: Iterable[int]
) -> np.ndarray:
    selected = tuple(int(value) for value in indices)
    if not selected:
        raise QCError("cannot average zero image volumes")
    total = np.zeros(tuple(int(value) for value in image.shape[:3]), dtype=np.float64)
    for index in selected:
        try:
            volume = np.asarray(image.dataobj[..., index], dtype=np.float64)
        except (OSError, ValueError, TypeError, IndexError) as error:
            raise QCError("cannot stream selected image volume") from error
        if not np.isfinite(volume).all():
            raise QCError("selected image volume contains non-finite data")
        total += volume
    total /= len(selected)
    return total


def _canonical_3d(data: np.ndarray, affine: np.ndarray, order: int = 1) -> np.ndarray:
    del order  # orientation only; no resampling is performed
    image = nib.Nifti1Image(np.asarray(data), affine)
    return np.asarray(nib.as_closest_canonical(image).dataobj, dtype=np.float64)


def _canonical_triplet(
    first: np.ndarray, second: np.ndarray, affine: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _canonical_3d(first, affine),
        _canonical_3d(second, affine),
        _canonical_3d(mask, affine),
    )


def _slice_location(mask: np.ndarray, fallback: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        finite = np.argwhere(np.isfinite(fallback) & (fallback != 0))
        coordinates = finite
    if coordinates.size == 0:
        return tuple(int(value // 2) for value in fallback.shape)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    return tuple(int(value) for value in ((lower + upper) // 2))


def _three_views(
    data: np.ndarray, location: tuple[int, int, int]
) -> tuple[tuple[np.ndarray, str], ...]:
    x, y, z = location
    return (
        (np.rot90(data[x, :, :]), "Sagittal"),
        (np.rot90(data[:, y, :]), "Coronal"),
        (np.rot90(data[:, :, z]), "Axial"),
    )


def _percentiles(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise QCError("display data have no finite foreground values")
    lower = float(np.percentile(finite, 1.0))
    upper = float(np.percentile(finite, 99.5))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise QCError("display percentiles are non-finite")
    if upper <= lower:
        upper = lower + max(abs(lower) * 1e-6, 1e-6)
    return lower, upper


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise QCError("difference data have no finite foreground values")
    limit = float(np.percentile(finite, 99.5))
    return limit if limit > 0 else 1e-6


def _load_numeric_table(path: Path, label: str) -> np.ndarray:
    try:
        values = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
    except (OSError, ValueError, TypeError) as error:
        raise QCError(f"cannot read {label}") from error
    if values.size == 0 or not np.isfinite(values).all():
        raise QCError(f"{label} must contain finite numeric values")
    return values


def _load_json(path: Path, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise QCError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except QCError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QCError(f"cannot read {label} JSON") from error
    if not isinstance(payload, dict):
        raise QCError(f"{label} must be a JSON object")
    _require_finite_json(payload, label)
    return payload


def _require_finite_json(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise QCError(f"{label} contains NaN or Infinity")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite_json(child, label)
    elif isinstance(value, list):
        for child in value:
            _require_finite_json(child, label)


def _classify(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise QCError("cSI values must be finite and positive")
    if value > _HIGH:
        return "high"
    if value >= _AMBIGUOUS:
        return "ambiguous"
    return "normal"


def _figure_basename(context: StageQCContext, figure_id: str) -> str:
    if figure_id not in FIGURE_FILENAMES:
        raise QCError(f"unknown figure ID: {figure_id!r}")
    name = FIGURE_FILENAMES[figure_id]
    if figure_id == "overview":
        name = name.replace("${subject_id}", context.stage_context.config.subject_id)
    return name


def _figure_path(context: StageQCContext, figure_id: str) -> Path:
    return context.output_directory / _figure_basename(context, figure_id)


def _non_overview_paths(context: StageQCContext) -> list[Path]:
    return [_figure_path(context, value) for value in NON_OVERVIEW_FIGURE_IDS]


def _require_regular(path: Path) -> None:
    _reject_traversal(path)
    _reject_symlink_components(path)
    try:
        details = path.lstat()
    except OSError as error:
        raise QCError(f"required input is not readable: {path.name}") from error
    if not stat.S_ISREG(details.st_mode):
        raise QCError(f"required input must be a regular file: {path.name}")
    if details.st_nlink != 1:
        raise QCError(
            f"required input must not be a hard-link; link count must be 1: {path.name}"
        )


def _open_regular_read(path: Path) -> int:
    _require_regular(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise QCError(f"cannot open required input: {path.name}") from error
    try:
        details = os.fstat(descriptor)
    except OSError as error:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise QCError(f"cannot inspect required input: {path.name}") from error
    if not stat.S_ISREG(details.st_mode):
        os.close(descriptor)
        raise QCError(f"required input must be regular: {path.name}")
    if details.st_nlink != 1:
        os.close(descriptor)
        raise QCError(
            f"required input must not be a hard-link; link count must be 1: {path.name}"
        )
    return descriptor


def _reject_traversal(path: Path) -> None:
    if ".." in path.parts:
        raise QCError("QC paths must not contain parent traversal")


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute(path)
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component):
            try:
                if stat.S_ISLNK(component.lstat().st_mode):
                    raise QCError("QC paths must not contain symbolic-link components")
            except OSError as error:
                raise QCError("cannot inspect QC path components") from error


def _relative_subject(
    path: Path, subject_root: Path, *, require_exists: bool = True
) -> Path:
    try:
        relative = path.resolve(strict=require_exists).relative_to(
            subject_root.resolve(strict=False)
        )
    except (OSError, ValueError) as error:
        raise QCError("scientific QC paths must remain within subject output") from error
    if not relative.parts:
        raise QCError("QC path must name an item within subject output")
    return relative


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _stat_signature(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(details.st_dev),
        int(details.st_ino),
        int(details.st_nlink),
        int(details.st_size),
        int(details.st_mtime_ns),
        int(details.st_ctime_ns),
    )


def _copy_descriptor_snapshot(
    descriptor: int,
    directory_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
) -> tuple[str, int, tuple[int, int]]:
    _require_safe_basename(name, "immutable QC snapshot")
    flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    output: int | None = None
    output_identity: tuple[int, int] | None = None
    digest = hashlib.sha256()
    try:
        output = os.open(
            name,
            flags,
            dir_fd=directory_descriptor,
        )
        details = os.fstat(output)
        if (
            not stat.S_ISREG(details.st_mode)
            or (int(details.st_dev), int(details.st_ino))
            != expected_identity
        ):
            raise QCError("immutable QC input snapshot is not a regular file")
        output_identity = (int(details.st_dev), int(details.st_ino))
        os.ftruncate(output, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(output, "wb", buffering=0, closefd=False) as stream:
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = stream.write(view)
                    if written is None or written <= 0:
                        raise OSError("short snapshot write")
                    view = view[written:]
            stream.flush()
            os.fsync(output)
        os.lseek(descriptor, 0, os.SEEK_SET)
        copied = digest.hexdigest()
        if (
            _hash_descriptor(descriptor) != copied
            or _hash_descriptor(output) != copied
        ):
            raise QCError("QC input changed while creating immutable snapshot")
        if output_identity is None:
            raise QCError("immutable QC input snapshot identity is unavailable")
        return copied, output, output_identity
    except BaseException as error:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError:
            pass
        if output is not None:
            try:
                os.ftruncate(output, 0)
                os.fsync(output)
            except OSError:
                pass
        if output_identity is not None:
            try:
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(current.st_mode)
                    and (int(current.st_dev), int(current.st_ino))
                    == output_identity
                ):
                    os.unlink(name, dir_fd=directory_descriptor)
            except OSError:
                pass
        if output is not None:
            try:
                os.close(output)
            except OSError:
                pass
        if isinstance(error, QCError):
            raise
        if isinstance(error, OSError):
            raise QCError(
                "cannot create immutable QC input snapshot"
            ) from error
        raise


def _discard_snapshot_descriptor(
    descriptor: int,
    directory_descriptor: int,
    name: str | None,
    identity: tuple[int, int] | None,
) -> None:
    try:
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
    except OSError:
        pass
    if name is not None and identity is not None:
        try:
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISREG(current.st_mode)
                and (int(current.st_dev), int(current.st_ino)) == identity
            ):
                os.unlink(name, dir_fd=directory_descriptor)
        except OSError:
            pass
    try:
        os.close(descriptor)
    except OSError:
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fd_identity(descriptor: int) -> tuple[int, int]:
    details = os.fstat(descriptor)
    return int(details.st_dev), int(details.st_ino)


def _recover_descriptor_stat(descriptor: int) -> os.stat_result:
    """Retry descriptor metadata through the independent ``stat(fd)`` API."""
    try:
        return os.fstat(descriptor)
    except OSError as first_error:
        try:
            return os.stat(descriptor)
        except OSError:
            raise first_error


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _not_number(value: str) -> bool:
    try:
        float(value)
        return False
    except ValueError:
        return True


def _reject_serialized_path_leaks(text: str) -> None:
    forbidden = ("/Users/", "file://", "http://", "https://", ".work")
    if any(value in text for value in forbidden):
        raise QCError("QC manifest contains a private path, URI, or work path")


__all__ = [
    "ACCEPTED_STAGE_NAMES",
    "DKI_DIRECT_KEYS",
    "DKI_KEYS",
    "DTI_KEYS",
    "FIGURE_FILENAMES",
    "FIGURE_IDS",
    "NODDI_KEYS",
    "QCError",
    "STAGE_FIGURES",
    "StageQCContext",
    "generate_all_qc",
    "generate_overview",
    "generate_stage_qc",
]
