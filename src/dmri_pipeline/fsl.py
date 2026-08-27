"""FSL discovery, validated command construction, and shell-free execution."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .config import PipelineConfig


_REQUIRED_TOOLS = (
    "topup",
    "applytopup",
    "bet",
    "fslmaths",
    "eddy_quad",
    "flirt",
    "fnirt",
    "invwarp",
    "applywarp",
)
_REQUIRED_RESOURCES = (
    "etc/flirtsch/b02b0.cnf",
    "etc/flirtsch/b02b0_1.cnf",
    "etc/flirtsch/FA_2_FMRIB58_1mm.cnf",
    "data/standard/FMRIB58_FA_1mm.nii.gz",
)
_MAX_EDDY_THREADS = 64
_MAX_RUNTIME_FILES = 256
_MAX_SCIENTIFIC_RUNTIME_FILES = 1024
_MAX_RUNTIME_DEPTH = 8
_MAX_RUNTIME_ENTRIES = 4096
_MAX_RUNTIME_FILE_BYTES = 64 * 1024 * 1024
_MAX_RUNTIME_TREE_BYTES = 256 * 1024 * 1024
_MAX_LAUNCHER_PREFIX_BYTES = 16 * 1024
_EDDY_QUAD_SCIENTIFIC_DEPENDENCIES = (
    "numpy",
    "nibabel",
    "matplotlib",
    "seaborn",
)


class FSLDiscoveryError(RuntimeError):
    """Raised when a complete, executable FSL installation cannot be found."""


class ExternalCommandError(RuntimeError):
    """Raised when an external command cannot launch or exits unsuccessfully."""

    def __init__(
        self,
        executable: Path,
        exit_code: int | None,
        log_path: Path,
        detail: str,
    ) -> None:
        self.executable = executable
        self.exit_code = exit_code
        self.log_path = log_path
        super().__init__(
            f"FSL command failed: executable={executable}, "
            f"exit_code={exit_code}, log={log_path}: {detail}"
        )


@dataclass(frozen=True)
class FSLInstallation:
    """Validated executable and resource paths for one FSL installation."""

    fsldir: Path
    topup: Path
    applytopup: Path
    bet: Path
    fslmaths: Path
    eddy: Path
    eddy_quad: Path
    flirt: Path
    fnirt: Path
    invwarp: Path
    applywarp: Path
    b02b0_config: Path
    b02b0_no_subsampling_config: Path
    fa_to_standard_config: Path
    standard_fa: Path
    _environment_items: tuple[tuple[str, str], ...] = field(repr=False)
    _runtime_material_items: tuple[tuple[str, Path], ...] = field(
        default=(), repr=False
    )

    @property
    def environment(self) -> dict[str, str]:
        """Return a fresh subject-local environment mapping."""
        return dict(self._environment_items)

    @property
    def env(self) -> dict[str, str]:
        """Return a fresh alias for the subject-local environment mapping."""
        return self.environment

    @property
    def runtime_material_files(self) -> dict[str, Path]:
        """Return material launcher backends keyed only by FSL-relative paths."""
        return dict(self._runtime_material_items)


@dataclass(frozen=True)
class FSLContext:
    """All paths and bounded settings needed by the FSL command builders.

    TOPUP, BET, and EDDY output prefixes must be extensionless because each
    tool materializes additional files by appending literal suffixes.
    """

    installation: FSLInstallation
    merged_b0: Path
    merged_b0_shape: tuple[int, ...]
    acqparams_topup: Path
    topup_prefix: Path
    topup_corrected_b0s: Path
    field_hz_prefix: Path
    applytopup_inputs: tuple[Path, ...]
    applytopup_indices: tuple[int, ...]
    applytopup_output: Path
    hifi_nodif: Path
    brain_prefix: Path
    gibbs_pa: Path
    cleaned_mask: Path
    acqparams_eddy: Path
    index_eddy: Path
    bvals: Path
    bvecs: Path
    eddy_prefix: Path
    eddy_threads: int
    eddy_quad_output: Path
    subject_fa: Path
    affine_fa: Path
    affine_matrix: Path
    nonlinear_fa: Path
    forward_warp: Path
    inverse_warp: Path
    atlas_labels: Path
    subject_atlas: Path

    def __post_init__(self) -> None:
        if not isinstance(self.installation, FSLInstallation):
            raise TypeError("installation must be an FSLInstallation")

        path_fields = (
            "merged_b0",
            "acqparams_topup",
            "topup_prefix",
            "topup_corrected_b0s",
            "field_hz_prefix",
            "applytopup_output",
            "hifi_nodif",
            "brain_prefix",
            "gibbs_pa",
            "cleaned_mask",
            "acqparams_eddy",
            "index_eddy",
            "bvals",
            "bvecs",
            "eddy_prefix",
            "eddy_quad_output",
            "subject_fa",
            "affine_fa",
            "affine_matrix",
            "nonlinear_fa",
            "forward_warp",
            "inverse_warp",
            "atlas_labels",
            "subject_atlas",
        )
        for name in path_fields:
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")

        shape = self.merged_b0_shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 4
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise ValueError(
                "merged_b0_shape must be a non-empty four-dimensional shape"
            )

        if (
            not isinstance(self.applytopup_inputs, tuple)
            or not self.applytopup_inputs
            or any(not isinstance(path, Path) for path in self.applytopup_inputs)
        ):
            raise ValueError("applytopup inputs must be a non-empty tuple of paths")
        if (
            not isinstance(self.applytopup_indices, tuple)
            or len(self.applytopup_indices) != len(self.applytopup_inputs)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 1
                for index in self.applytopup_indices
            )
        ):
            raise ValueError(
                "applytopup indices must be one-based integers matching the inputs"
            )

        if (
            isinstance(self.eddy_threads, bool)
            or not isinstance(self.eddy_threads, int)
            or not 1 <= self.eddy_threads <= _MAX_EDDY_THREADS
        ):
            raise ValueError(
                f"eddy_threads must be an integer from 1 to {_MAX_EDDY_THREADS}"
            )

        all_context_paths = [
            *(getattr(self, name) for name in path_fields),
            *self.applytopup_inputs,
        ]
        for path in all_context_paths:
            if ".." in path.parts:
                raise ValueError(
                    f"FSL context paths must not contain '..' components: {path}"
                )
        for name in ("topup_prefix", "brain_prefix", "eddy_prefix"):
            value = str(getattr(self, name)).lower()
            if value.endswith(".nii") or value.endswith(".nii.gz"):
                raise ValueError(
                    f"{name} must be extensionless because FSL appends "
                    "literal output sidecar suffixes"
                )

        _validate_context_path_identities(self)

    @property
    def fsldir(self) -> Path:
        return self.installation.fsldir

    @property
    def rotated_bvecs(self) -> Path:
        return Path(f"{self.eddy_prefix}.eddy_rotated_bvecs")

    @property
    def field_hz_image(self) -> Path:
        value = str(self.field_hz_prefix)
        if value.endswith(".nii") or value.endswith(".nii.gz"):
            return self.field_hz_prefix
        return Path(value + ".nii.gz")


def _validate_context_path_identities(context: FSLContext) -> None:
    installation = context.installation
    protected_input_groups = (
        _nifti_identities(context.merged_b0),
        _exact_identities(context.acqparams_topup),
        *(_nifti_identities(path) for path in context.applytopup_inputs),
        _nifti_identities(context.gibbs_pa),
        _nifti_identities(context.cleaned_mask),
        _exact_identities(context.acqparams_eddy),
        _exact_identities(context.index_eddy),
        _exact_identities(context.bvals),
        _exact_identities(context.bvecs),
        _nifti_identities(context.subject_fa),
        _nifti_identities(context.atlas_labels),
        _exact_identities(installation.topup),
        _exact_identities(installation.applytopup),
        _exact_identities(installation.bet),
        _exact_identities(installation.fslmaths),
        _exact_identities(installation.eddy),
        _exact_identities(installation.eddy_quad),
        _exact_identities(installation.flirt),
        _exact_identities(installation.fnirt),
        _exact_identities(installation.invwarp),
        _exact_identities(installation.applywarp),
        _exact_identities(installation.b02b0_config),
        _exact_identities(installation.b02b0_no_subsampling_config),
        _exact_identities(installation.fa_to_standard_config),
        _nifti_identities(installation.standard_fa),
    )
    protected_inputs = frozenset().union(*protected_input_groups)

    topup_stem = _nifti_stem(context.topup_prefix)
    brain_stem = _nifti_stem(context.brain_prefix)
    eddy_stem = _nifti_stem(context.eddy_prefix)
    output_groups = (
        (
            "topup_prefix",
            _exact_identities(context.topup_prefix)
            | _nifti_identities(Path(f"{topup_stem}_fieldcoef"))
            | _exact_identities(Path(f"{topup_stem}_movpar.txt")),
        ),
        ("topup_corrected_b0s", _nifti_identities(context.topup_corrected_b0s)),
        ("field_hz_prefix", _nifti_identities(context.field_hz_prefix)),
        ("applytopup_output", _nifti_identities(context.applytopup_output)),
        ("hifi_nodif", _nifti_identities(context.hifi_nodif)),
        (
            "brain_prefix",
            _nifti_identities(context.brain_prefix)
            | _nifti_identities(Path(f"{brain_stem}_mask")),
        ),
        ("eddy_prefix", _eddy_output_identities(eddy_stem)),
        ("eddy_quad_output", _exact_identities(context.eddy_quad_output)),
        ("affine_fa", _nifti_identities(context.affine_fa)),
        ("affine_matrix", _exact_identities(context.affine_matrix)),
        ("nonlinear_fa", _nifti_identities(context.nonlinear_fa)),
        ("forward_warp", _nifti_identities(context.forward_warp)),
        ("inverse_warp", _nifti_identities(context.inverse_warp)),
        ("subject_atlas", _nifti_identities(context.subject_atlas)),
    )

    for name, identities in output_groups:
        collisions = protected_inputs.intersection(identities)
        if collisions:
            paths = ", ".join(sorted(map(str, collisions)))
            raise ValueError(
                f"FSL output {name} would overwrite a raw or upstream input: "
                f"{paths}"
            )

    occupied: dict[object, str] = {}
    for name, identities in output_groups:
        collisions = {
            identity: occupied[identity]
            for identity in identities
            if identity in occupied
        }
        if collisions:
            other_names = ", ".join(sorted(set(collisions.values())))
            paths = ", ".join(sorted(map(str, collisions)))
            raise ValueError(
                f"FSL output path collision between {name} and {other_names}: "
                f"{paths}"
            )
        for identity in identities:
            occupied[identity] = name


def _eddy_output_identities(stem: Path) -> frozenset[object]:
    identities = set(_nifti_identities(stem))
    for suffix in (
        ".eddy_cnr_maps",
        ".eddy_residuals",
        ".eddy_outlier_map",
        ".eddy_outlier_n_stdev_map",
        ".eddy_outlier_n_sqr_stdev_map",
    ):
        identities.update(_nifti_identities(Path(f"{stem}{suffix}")))
    for suffix in (
        ".eddy_rotated_bvecs",
        ".eddy_parameters",
        ".eddy_movement_rms",
        ".eddy_restricted_movement_rms",
        ".eddy_outlier_report",
        ".eddy_command_txt",
        ".eddy_values_of_all_input_parameters",
        ".eddy_post_eddy_shell_alignment_parameters",
        ".eddy_post_eddy_shell_PE_translation_parameters",
    ):
        identities.update(_exact_identities(Path(f"{stem}{suffix}")))
    return frozenset(identities)


def _nifti_identities(path: Path) -> frozenset[object]:
    stem = _nifti_stem(path)
    identities: set[object] = set()
    for candidate in (
        path,
        stem,
        Path(f"{stem}.nii"),
        Path(f"{stem}.nii.gz"),
    ):
        identities.update(_candidate_identities(candidate))
    return frozenset(identities)


def _exact_identities(path: Path) -> frozenset[object]:
    return _candidate_identities(path)


def _nifti_stem(path: Path) -> Path:
    value = str(path)
    if value.endswith(".nii.gz"):
        return Path(value[:-7])
    if value.endswith(".nii"):
        return Path(value[:-4])
    return path


def _resolved_identity(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot resolve FSL context path safely: {path}") from error


def _candidate_identities(path: Path) -> frozenset[object]:
    resolved = _resolved_identity(path)
    identities: set[object] = {resolved}
    try:
        metadata = resolved.stat()
    except FileNotFoundError:
        return frozenset(identities)
    except OSError as error:
        raise ValueError(
            f"Cannot inspect FSL context path identity safely: {path}"
        ) from error
    identities.add(("inode", metadata.st_dev, metadata.st_ino))
    return frozenset(identities)


def discover_fsl(config: PipelineConfig) -> FSLInstallation:
    """Discover and validate FSL without changing the process environment.

    Direct library callers may omit ``DMRI_EXPECTED_FSL_VERSION`` to perform
    capability-only discovery. Public wrappers always export it and therefore
    enforce the exact configured version, including for YAML-selected FSL.
    """
    configured = getattr(config, "fsldir", None)
    source: str
    if configured is not None:
        candidate = Path(configured).expanduser().resolve(strict=False)
        source = "explicit tools.fsldir"
    else:
        process_fsldir = os.environ.get("FSLDIR")
        if process_fsldir:
            candidate = Path(process_fsldir).expanduser().resolve(strict=False)
            source = "process FSLDIR"
        else:
            topup_on_path = shutil.which("topup")
            if topup_on_path is None:
                raise FSLDiscoveryError(
                    "FSL was not found: configure tools.fsldir, set FSLDIR, "
                    "or put topup on PATH"
                )
            candidate = Path(topup_on_path).resolve(strict=True).parent.parent
            source = "topup on PATH"

    try:
        return _validate_installation(candidate)
    except FSLDiscoveryError as error:
        raise FSLDiscoveryError(
            f"Invalid FSL installation from {source} ({candidate}): {error}"
        ) from error


def _validate_installation(fsldir: Path) -> FSLInstallation:
    if not fsldir.is_dir():
        raise FSLDiscoveryError("FSLDIR is not a directory")

    _validate_expected_fsl_version(fsldir)

    tools: dict[str, Path] = {}
    for name in _REQUIRED_TOOLS:
        executable = fsldir / "bin" / name
        _require_executable(executable, discovery=True)
        tools[name] = executable

    openmp = fsldir / "bin" / "eddy_openmp"
    eddy_command = fsldir / "bin" / "eddy"
    if _is_executable_regular_file(openmp):
        eddy = openmp
    elif _is_executable_regular_file(eddy_command):
        launcher = _read_stable_runtime_prefix(
            eddy_command, "selected EDDY launcher"
        )
        if launcher.startswith(b"#!"):
            if launcher.splitlines()[0] != b"#!/usr/bin/env fslpython":
                raise FSLDiscoveryError(
                    "selected script EDDY must use the FSL fslpython "
                    "interpreter"
                )
            cpu_backend = fsldir / "bin" / "eddy_cpu"
            if not _is_executable_regular_file(cpu_backend):
                raise FSLDiscoveryError(
                    "selected EDDY launcher is missing executable CPU backend: "
                    "eddy_cpu"
                )
            eddy = cpu_backend
        else:
            eddy = eddy_command
    else:
        raise FSLDiscoveryError(
            f"missing executable regular file: {openmp.name} or {eddy_command.name}"
        )

    resources: dict[str, Path] = {}
    for relative in _REQUIRED_RESOURCES:
        resource = fsldir / relative
        if not resource.is_file():
            raise FSLDiscoveryError(f"missing required file: {resource}")
        resources[relative] = resource

    environment = os.environ.copy()
    existing_path = environment.get("PATH", "")
    bin_path = str(fsldir / "bin")
    environment["PATH"] = (
        bin_path + os.pathsep + existing_path if existing_path else bin_path
    )
    environment["FSLDIR"] = str(fsldir)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    runtime_material = _discover_runtime_material_files(fsldir, eddy)

    return FSLInstallation(
        fsldir=fsldir,
        topup=tools["topup"],
        applytopup=tools["applytopup"],
        bet=tools["bet"],
        fslmaths=tools["fslmaths"],
        eddy=eddy,
        eddy_quad=tools["eddy_quad"],
        flirt=tools["flirt"],
        fnirt=tools["fnirt"],
        invwarp=tools["invwarp"],
        applywarp=tools["applywarp"],
        b02b0_config=resources["etc/flirtsch/b02b0.cnf"],
        b02b0_no_subsampling_config=resources[
            "etc/flirtsch/b02b0_1.cnf"
        ],
        fa_to_standard_config=resources[
            "etc/flirtsch/FA_2_FMRIB58_1mm.cnf"
        ],
        standard_fa=resources["data/standard/FMRIB58_FA_1mm.nii.gz"],
        _environment_items=tuple(sorted(environment.items())),
        _runtime_material_items=runtime_material,
    )


def _validate_expected_fsl_version(fsldir: Path) -> None:
    expected = os.environ.get("DMRI_EXPECTED_FSL_VERSION")
    if expected is None:
        return
    if not expected:
        raise FSLDiscoveryError("DMRI_EXPECTED_FSL_VERSION must not be empty")
    version_file = fsldir / "etc" / "fslversion"
    try:
        contents = _read_stable_runtime_prefix(
            version_file, "FSL version file", require_complete=True
        ).decode("utf-8")
    except (UnicodeError, FSLDiscoveryError) as error:
        raise FSLDiscoveryError(
            "cannot read a valid FSL version file: etc/fslversion"
        ) from error
    lines = contents.splitlines()
    actual = lines[0].strip() if lines else ""
    if not actual:
        raise FSLDiscoveryError("FSL version file is empty")
    if actual != expected:
        raise FSLDiscoveryError(
            f"FSL version mismatch: expected {expected}, found {actual}"
        )


def _discover_runtime_material_files(
    fsldir: Path, selected_eddy: Path
) -> tuple[tuple[str, Path], ...]:
    material: dict[str, Path] = {}

    def add(relative: str, *, executable: bool = False) -> Path:
        if relative in material:
            return material[relative]
        path = fsldir / relative
        _validate_readable_runtime_file(path, relative)
        if executable and not os.access(path, os.X_OK):
            raise FSLDiscoveryError(
                f"material FSL runtime file is not executable: {relative}"
            )
        material[relative] = path
        return path

    for relative in (
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
    ):
        add(relative, executable=True)

    fslpython = add("bin/fslpython", executable=True)
    fslpython_launcher = _read_stable_runtime_prefix(
        fslpython,
        "bin/fslpython",
        require_complete=True,
    )
    _validate_fslpython_launcher(fslpython_launcher)

    eddy_quad_prefix = _read_stable_runtime_prefix(
        fsldir / "bin" / "eddy_quad", "bin/eddy_quad launcher"
    )
    _validate_eddy_quad_launcher(eddy_quad_prefix, fsldir)

    eddy_is_launcher = False
    if selected_eddy.name == "eddy":
        launcher = _read_stable_runtime_prefix(
            selected_eddy, "selected EDDY launcher"
        )
        if launcher.startswith(b"#!"):
            if launcher.splitlines()[0] != b"#!/usr/bin/env fslpython":
                raise FSLDiscoveryError(
                    "selected script EDDY must use the FSL fslpython "
                    "interpreter"
                )
            eddy_is_launcher = True
            add("bin/eddy_cpu", executable=True)
            add("bin/find_cuda_exe", executable=True)
            candidates = sorted(
                (path for path in (fsldir / "bin").glob("eddy_cuda*")),
                key=lambda path: path.name,
            )
            if len(candidates) > 16:
                raise FSLDiscoveryError(
                    "too many candidate EDDY CUDA backends to fingerprint safely"
                )
            for candidate in candidates:
                try:
                    metadata = candidate.lstat()
                except OSError as error:
                    raise FSLDiscoveryError(
                        f"cannot inspect EDDY backend: {candidate.name}"
                    ) from error
                if stat.S_ISREG(metadata.st_mode) and os.access(candidate, os.X_OK):
                    add(f"bin/{candidate.name}", executable=True)
                else:
                    raise FSLDiscoveryError(
                        f"candidate EDDY backend must be an executable regular "
                        f"file: {candidate.name}"
                    )

    quad_roots: list[Path] = []
    for candidate in sorted(
        (fsldir / "lib").glob("python*/site-packages/eddy_qc"),
        key=lambda path: path.as_posix(),
    ):
        python_root = candidate.parents[1]
        try:
            python_metadata = python_root.lstat()
        except OSError as error:
            raise FSLDiscoveryError(
                "cannot inspect the eddy_qc Python runtime root"
            ) from error
        if stat.S_ISLNK(python_metadata.st_mode):
            continue
        if not stat.S_ISDIR(python_metadata.st_mode):
            raise FSLDiscoveryError(
                "eddy_qc Python runtime root must be a non-symbolic directory"
            )
        try:
            site_metadata = candidate.parent.lstat()
        except OSError as error:
            raise FSLDiscoveryError(
                "cannot inspect the eddy_qc site-packages directory"
            ) from error
        if not stat.S_ISDIR(site_metadata.st_mode):
            raise FSLDiscoveryError(
                "eddy_qc site-packages must be a non-symbolic directory"
            )
        quad_roots.append(candidate)
    if len(quad_roots) != 1:
        raise FSLDiscoveryError(
            "expected exactly one installed eddy_qc runtime package"
        )
    quad_root = quad_roots[0]
    python_dir = quad_root.parents[1].name
    site_packages = quad_root.parent
    add(f"bin/{python_dir}", executable=True)
    for relative in (
        f"lib/{python_dir}/site-packages/fsl/scripts/remove_ext.py",
        f"lib/{python_dir}/site-packages/fsl/scripts/imtest.py",
        f"lib/{python_dir}/site-packages/fsl/utils/path.py",
    ):
        add(relative)
    python_alias = fsldir / "bin" / "python"
    interpreter = _resolve_stable_runtime_interpreter(
        python_alias, fsldir
    )
    interpreter_relative = interpreter.relative_to(fsldir).as_posix()
    add(interpreter_relative, executable=True)
    material["bin/python"] = interpreter
    if eddy_is_launcher:
        for relative in (
            f"lib/{python_dir}/site-packages/fsl/utils/run.py",
            f"lib/{python_dir}/site-packages/fsl/base/find_cuda_exe.py",
        ):
            add(relative)
    for path in _bounded_runtime_tree(
        site_packages / "fsl",
        label="fslpy runtime package",
        suffixes=frozenset({".py"}),
    ):
        relative = path.relative_to(fsldir).as_posix()
        material[relative] = path
    for path in _bounded_runtime_tree(
        quad_root,
        label="eddy_qc runtime package",
    ):
        relative = path.relative_to(fsldir).as_posix()
        material[relative] = path
    for dependency in _EDDY_QUAD_SCIENTIFIC_DEPENDENCIES:
        dependency_root = site_packages / dependency
        dependency_files = _bounded_runtime_tree(
            dependency_root,
            label=f"{dependency} EDDY QUAD runtime package",
            max_files=_MAX_SCIENTIFIC_RUNTIME_FILES,
        )
        required_module = dependency_root / "__init__.py"
        if required_module not in dependency_files:
            raise FSLDiscoveryError(
                f"missing material {dependency} EDDY QUAD module"
            )
        for path in dependency_files:
            relative = path.relative_to(fsldir).as_posix()
            material[relative] = path
    for distribution in (
        "eddy_qc",
        *_EDDY_QUAD_SCIENTIFIC_DEPENDENCIES,
    ):
        for path in _unique_distribution_runtime_files(
            site_packages, distribution
        ):
            relative = path.relative_to(fsldir).as_posix()
            material[relative] = path
    for required in (
        quad_root / "__init__.py",
        quad_root / "scripts" / "eddy_quad.py",
        quad_root / "QUAD" / "quad.py",
    ):
        relative = required.relative_to(fsldir).as_posix()
        if relative not in material:
            raise FSLDiscoveryError(
                f"missing material EDDY QUAD module: {relative}"
            )
    return tuple(sorted(material.items()))


def _unique_distribution_runtime_files(
    site_packages: Path, distribution: str
) -> tuple[Path, ...]:
    metadata_files = sorted(
        site_packages.glob(f"{distribution}-*.dist-info/METADATA"),
        key=lambda path: path.as_posix(),
    )
    if len(metadata_files) != 1:
        raise FSLDiscoveryError(
            f"expected exactly one {distribution} distribution metadata file"
        )
    files = _bounded_runtime_tree(
        metadata_files[0].parent,
        label=f"{distribution} distribution metadata",
    )
    if metadata_files[0] not in files:
        raise FSLDiscoveryError(
            f"missing material {distribution} distribution metadata"
        )
    return files


def _read_stable_runtime_prefix(
    path: Path,
    label: str,
    *,
    require_complete: bool = False,
) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise FSLDiscoveryError(f"cannot inspect {label}") from error
    maximum_size = (
        _MAX_LAUNCHER_PREFIX_BYTES
        if require_complete
        else _MAX_RUNTIME_FILE_BYTES
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > maximum_size
    ):
        raise FSLDiscoveryError(
            f"{label} must be a nonempty bounded regular file"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FSLDiscoveryError(f"cannot open {label}") from error
    try:
        opened_before = os.fstat(descriptor)
        prefix = os.read(descriptor, _MAX_LAUNCHER_PREFIX_BYTES)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise FSLDiscoveryError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise FSLDiscoveryError(f"{label} changed during inspection") from error
    expected = _runtime_file_identity(before)
    if (
        _runtime_file_identity(opened_before) != expected
        or _runtime_file_identity(opened_after) != expected
        or _runtime_file_identity(after) != expected
    ):
        raise FSLDiscoveryError(f"{label} changed during inspection")
    if not prefix:
        raise FSLDiscoveryError(f"{label} is empty")
    if require_complete and len(prefix) != before.st_size:
        raise FSLDiscoveryError(f"{label} changed during inspection")
    return prefix


def _validate_fslpython_launcher(launcher: bytes) -> None:
    targets = (
        rb'"\$\{FSLDIR\}/bin/python"',
        rb"\$\{FSLDIR\}/bin/python",
        rb'"\$FSLDIR/bin/python"',
        rb"\$FSLDIR/bin/python",
    )
    installed_fsldir_guard = (
        rb'\nif \[ "\$FSLDIR" = "" \] ; then\n'
        rb'  echo "FSLDIR has not been set!"\n'
        rb"  exit 1\n"
        rb"fi\n\n"
    )
    pattern = (
        rb"\A#!/bin/sh\n(?:"
        + installed_fsldir_guard
        + rb")?exec[ \t]+(?:"
        + rb"|".join(targets)
        + rb')[ \t]+"\$@"[ \t]*\n?\Z'
    )
    if re.fullmatch(pattern, launcher) is None:
        raise FSLDiscoveryError(
            "bin/fslpython must be a launcher for FSLDIR/bin/python"
        )


def _validate_eddy_quad_launcher(prefix: bytes, fsldir: Path) -> None:
    targets = (
        re.escape(os.fsencode(fsldir / "bin" / "python")),
        rb'"\$\{FSLDIR\}/bin/python"',
        rb"\$\{FSLDIR\}/bin/python",
        rb'"\$FSLDIR/bin/python"',
        rb"\$FSLDIR/bin/python",
    )
    pattern = (
        rb"\A#!/bin/sh\n'''exec'[ \t]+(?:"
        + rb"|".join(targets)
        + rb')[ \t]+"\$0"[ \t]+"\$@"\n'
        + rb"' '''\n"
    )
    if re.match(pattern, prefix) is None:
        raise FSLDiscoveryError(
            "bin/eddy_quad launcher must execute FSL bin/python"
        )


def _resolve_stable_runtime_interpreter(
    alias: Path, fsldir: Path
) -> Path:
    try:
        before = alias.lstat()
    except OSError as error:
        raise FSLDiscoveryError(
            "cannot inspect the EDDY QUAD Python interpreter alias"
        ) from error
    if not (
        stat.S_ISLNK(before.st_mode) or stat.S_ISREG(before.st_mode)
    ):
        raise FSLDiscoveryError(
            "EDDY QUAD Python interpreter alias is unsafe"
        )
    try:
        interpreter = alias.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FSLDiscoveryError(
            "EDDY QUAD Python interpreter alias is broken"
        ) from error
    if interpreter.parent != fsldir / "bin":
        raise FSLDiscoveryError(
            "EDDY QUAD Python interpreter must stay inside FSL bin"
        )
    _validate_readable_runtime_file(
        interpreter,
        "EDDY QUAD Python interpreter",
        require_nonempty=True,
    )
    if not os.access(interpreter, os.X_OK):
        raise FSLDiscoveryError(
            "EDDY QUAD Python interpreter is not executable"
        )
    try:
        after = alias.lstat()
        resolved_after = alias.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FSLDiscoveryError(
            "EDDY QUAD Python interpreter alias changed during discovery"
        ) from error
    if (
        _runtime_file_identity(after) != _runtime_file_identity(before)
        or resolved_after != interpreter
    ):
        raise FSLDiscoveryError(
            "EDDY QUAD Python interpreter alias changed during discovery"
        )
    return interpreter


def _bounded_runtime_tree(
    root: Path,
    *,
    label: str,
    suffixes: frozenset[str] | None = None,
    max_files: int = _MAX_RUNTIME_FILES,
    max_file_bytes: int = _MAX_RUNTIME_FILE_BYTES,
    max_tree_bytes: int = _MAX_RUNTIME_TREE_BYTES,
) -> tuple[Path, ...]:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise FSLDiscoveryError(f"cannot inspect {label}") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise FSLDiscoveryError(f"{label} must be a non-symbolic directory")
    entries: list[Path] = []
    try:
        for path in root.rglob("*"):
            entries.append(path)
            if len(entries) > _MAX_RUNTIME_ENTRIES:
                raise FSLDiscoveryError(
                    f"{label} exceeds the bounded entry limit"
                )
    except OSError as error:
        raise FSLDiscoveryError(f"cannot traverse {label}") from error
    files: list[Path] = []
    total_bytes = 0
    for path in sorted(entries, key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(root)
        if len(relative.parts) > _MAX_RUNTIME_DEPTH:
            raise FSLDiscoveryError(f"{label} nesting is unsafe")
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FSLDiscoveryError(
                f"cannot inspect {label} entry: {relative}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise FSLDiscoveryError(
                f"{label} contains a symbolic link: {relative}"
            )
        if "__pycache__" in relative.parts:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise FSLDiscoveryError(
                f"{label} contains an unsafe entry: {relative}"
            )
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        _validate_readable_runtime_file(
            path,
            f"{label} entry {relative}",
            max_bytes=max_file_bytes,
        )
        files.append(path)
        total_bytes += metadata.st_size
        if len(files) > max_files:
            raise FSLDiscoveryError(
                f"{label} exceeds the bounded file limit of {max_files}"
            )
        if total_bytes > max_tree_bytes:
            raise FSLDiscoveryError(
                f"{label} exceeds the bounded byte limit"
            )
    return tuple(files)


def _runtime_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_readable_runtime_file(
    path: Path,
    label: str,
    *,
    max_bytes: int = _MAX_RUNTIME_FILE_BYTES,
    require_nonempty: bool = False,
) -> None:
    try:
        before = path.lstat()
    except OSError as error:
        raise FSLDiscoveryError(
            f"cannot inspect material FSL runtime file: {label}"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        raise FSLDiscoveryError(
            f"material FSL runtime file must be regular, not symbolic: {label}"
        )
    if before.st_size > max_bytes:
        raise FSLDiscoveryError(
            f"material FSL runtime file exceeds the {max_bytes}-byte "
            f"size limit: {label}"
        )
    if require_nonempty and before.st_size <= 0:
        raise FSLDiscoveryError(
            f"material FSL runtime file must be nonempty: {label}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FSLDiscoveryError(
            f"material FSL runtime file is not readable: {label}"
        ) from error
    try:
        opened_before = os.fstat(descriptor)
        os.read(descriptor, 1)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise FSLDiscoveryError(
            f"cannot read material FSL runtime file: {label}"
        ) from error
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise FSLDiscoveryError(
            f"material FSL runtime file changed during discovery: {label}"
        ) from error

    expected = _runtime_file_identity(before)
    if (
        _runtime_file_identity(opened_before) != expected
        or _runtime_file_identity(opened_after) != expected
        or _runtime_file_identity(after) != expected
    ):
        raise FSLDiscoveryError(
            f"material FSL runtime file changed during discovery: {label}"
        )


def build_topup_command(context: FSLContext) -> list[str]:
    """Build TOPUP with the safe config for the merged b0 spatial shape."""
    _validated_context(context)
    config = (
        context.installation.b02b0_no_subsampling_config
        if any(dimension % 2 for dimension in context.merged_b0_shape[:3])
        else context.installation.b02b0_config
    )
    return [
        str(context.installation.topup),
        f"--imain={context.merged_b0}",
        f"--datain={context.acqparams_topup}",
        f"--config={config}",
        f"--out={context.topup_prefix}",
        f"--iout={context.topup_corrected_b0s}",
        f"--fout={context.field_hz_prefix}",
    ]


def build_applytopup_command(context: FSLContext) -> list[str]:
    """Build APPLYTOPUP using explicit one-based acquisition rows."""
    _validated_context(context)
    return [
        str(context.installation.applytopup),
        "--imain=" + ",".join(map(str, context.applytopup_inputs)),
        f"--datain={context.acqparams_topup}",
        "--inindex=" + ",".join(map(str, context.applytopup_indices)),
        f"--topup={context.topup_prefix}",
        "--method=jac",
        f"--out={context.applytopup_output}",
    ]


def build_topup_mean_command(context: FSLContext) -> list[str]:
    """Build the corrected mean-b0 calculation used as BET input."""
    _validated_context(context)
    return [
        str(context.installation.fslmaths),
        str(context.topup_corrected_b0s),
        "-Tmean",
        str(context.hifi_nodif),
    ]


def build_bet_command(context: FSLContext) -> list[str]:
    """Build robust BET brain extraction with the accepted fixed settings."""
    _validated_context(context)
    return [
        str(context.installation.bet),
        str(context.hifi_nodif),
        str(context.brain_prefix),
        "-R",
        "-f",
        "0.25",
        "-g",
        "0",
        "-m",
    ]


def build_eddy_command(context: FSLContext) -> list[str]:
    """Build CPU EDDY using corrected PA data and original gradients."""
    _validated_context(context)
    return [
        str(context.installation.eddy),
        f"--imain={context.gibbs_pa}",
        f"--mask={context.cleaned_mask}",
        f"--acqp={context.acqparams_eddy}",
        f"--index={context.index_eddy}",
        f"--bvecs={context.bvecs}",
        f"--bvals={context.bvals}",
        f"--topup={context.topup_prefix}",
        "--repol",
        "--cnr_maps",
        "--residuals",
        "--data_is_shelled",
        f"--nthr={context.eddy_threads}",
        f"--out={context.eddy_prefix}",
    ]


def build_eddy_quad_command(context: FSLContext) -> list[str]:
    """Build EDDY QUAD using EDDY-rotated b-vectors."""
    _validated_context(context)
    return [
        str(context.installation.eddy_quad),
        str(context.eddy_prefix),
        "-idx",
        str(context.index_eddy),
        "-par",
        str(context.acqparams_eddy),
        "-m",
        str(context.cleaned_mask),
        "-b",
        str(context.bvals),
        "-g",
        str(context.rotated_bvecs),
        "-f",
        str(context.field_hz_image),
        "-o",
        str(context.eddy_quad_output),
    ]


def build_jhu_commands(context: FSLContext) -> list[list[str]]:
    """Build affine, nonlinear, inverse, and nearest-label registration."""
    _validated_context(context)
    installation = context.installation
    return [
        [
            str(installation.flirt),
            "-in",
            str(context.subject_fa),
            "-ref",
            str(installation.standard_fa),
            "-out",
            str(context.affine_fa),
            "-omat",
            str(context.affine_matrix),
            "-dof",
            "12",
            "-cost",
            "corratio",
        ],
        [
            str(installation.fnirt),
            f"--in={context.subject_fa}",
            f"--ref={installation.standard_fa}",
            f"--aff={context.affine_matrix}",
            f"--cout={context.forward_warp}",
            f"--iout={context.nonlinear_fa}",
            f"--config={installation.fa_to_standard_config}",
        ],
        [
            str(installation.invwarp),
            f"--warp={context.forward_warp}",
            f"--ref={context.subject_fa}",
            f"--out={context.inverse_warp}",
        ],
        [
            str(installation.applywarp),
            f"--in={context.atlas_labels}",
            f"--ref={context.subject_fa}",
            f"--warp={context.inverse_warp}",
            f"--out={context.subject_atlas}",
            "--interp=nn",
        ],
    ]


def run_fsl_command(
    argv: Sequence[str],
    log_path: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Execute an argv directly and append deterministic combined output."""
    command = _validated_argv(argv)
    executable = Path(command[0])
    path = Path(log_path)
    _require_executable_for_run(executable, path)
    if ".." in path.parts:
        raise ExternalCommandError(
            executable,
            None,
            path,
            "stage log path must not contain '..' components",
        )
    if path.is_symlink():
        raise ExternalCommandError(
            executable, None, path, "refusing symlink stage log path"
        )
    environment = _validated_environment(env, executable, path)

    try:
        descriptor = _open_append_no_symlink(path)
    except OSError as error:
        detail = (
            "refusing symlink stage log path or parent component"
            if error.errno == errno.ELOOP
            else f"cannot open stage log: {error}"
        )
        raise ExternalCommandError(executable, None, path, detail) from error

    with os.fdopen(
        descriptor,
        "a",
        encoding="utf-8",
        errors="replace",
    ) as log:
        log.write("=== FSL COMMAND START ===\n")
        log.write("ARGV_JSON=" + json.dumps(list(command), ensure_ascii=False) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                shell=False,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, TypeError, ValueError) as error:
            log.write(f"LAUNCH_ERROR={type(error).__name__}: {error}\n")
            log.write("EXIT_CODE=not-launched\n")
            log.write("=== FSL COMMAND END ===\n")
            log.flush()
            raise ExternalCommandError(
                executable,
                None,
                path,
                f"could not launch executable: {error}",
            ) from error

        log.write(f"EXIT_CODE={completed.returncode}\n")
        log.write("=== FSL COMMAND END ===\n")
        log.flush()

    if completed.returncode != 0:
        raise ExternalCommandError(
            executable,
            completed.returncode,
            path,
            "see the stage log for combined stdout and stderr",
        )
    return completed


def _validated_context(context: FSLContext) -> FSLContext:
    if not isinstance(context, FSLContext):
        raise TypeError("context must be an FSLContext")
    return context


def _is_executable_regular_file(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and os.access(path, os.X_OK)


def _require_executable(path: Path, *, discovery: bool) -> None:
    if _is_executable_regular_file(path):
        return
    if discovery:
        raise FSLDiscoveryError(f"missing executable regular file: {path}")
    raise AssertionError("unreachable")


def _validated_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise TypeError("argv must be a non-string sequence of strings")
    if not argv:
        raise ValueError("argv must not be empty")
    if any(not isinstance(argument, str) for argument in argv):
        raise TypeError("argv arguments must all be strings")
    if not argv[0]:
        raise ValueError("argv[0] must not be empty")
    return tuple(argv)


def _require_executable_for_run(executable: Path, log_path: Path) -> None:
    try:
        mode = executable.stat().st_mode
    except FileNotFoundError as error:
        raise ExternalCommandError(
            executable, None, log_path, "executable does not exist"
        ) from error
    except OSError as error:
        raise ExternalCommandError(
            executable, None, log_path, f"cannot inspect executable: {error}"
        ) from error
    if not stat.S_ISREG(mode):
        raise ExternalCommandError(
            executable, None, log_path, "executable is not a regular file"
        )
    if not os.access(executable, os.X_OK):
        raise ExternalCommandError(
            executable, None, log_path, "executable is not executable"
        )


def _open_append_no_symlink(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError(
            errno.ENOTSUP,
            "platform lacks no-follow directory traversal support",
        )

    absolute = path if path.is_absolute() else Path.cwd() / path
    parts = absolute.parts
    if len(parts) < 2 or not absolute.name:
        raise OSError(errno.EINVAL, "stage log path must name a file")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    directory_fd = os.open(absolute.anchor, directory_flags)
    try:
        for component in parts[1:-1]:
            try:
                metadata = os.stat(
                    component,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                metadata = os.stat(
                    component,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )

            if stat.S_ISLNK(metadata.st_mode):
                raise OSError(
                    errno.ELOOP,
                    f"symlink log parent component: {component}",
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError(
                    errno.ENOTDIR,
                    f"log parent component is not a directory: {component}",
                )

            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd

        try:
            existing = os.stat(
                parts[-1],
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _require_safe_log_metadata(existing)

        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        descriptor = os.open(parts[-1], flags, 0o600, dir_fd=directory_fd)
        try:
            _require_safe_log_metadata(os.fstat(descriptor))
        except OSError:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(directory_fd)


def _require_safe_log_metadata(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise OSError(errno.ELOOP, "refusing symlink stage log")
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(errno.EINVAL, "stage log must be a regular file")
    if metadata.st_nlink != 1:
        raise OSError(
            errno.EMLINK,
            f"refusing hard-linked stage log with {metadata.st_nlink} links",
        )


def _validated_environment(
    env: Mapping[str, str],
    executable: Path,
    log_path: Path,
) -> dict[str, str]:
    if not isinstance(env, Mapping):
        raise ExternalCommandError(
            executable,
            None,
            log_path,
            "environment must be a mapping of strings to strings",
        )
    try:
        items = tuple(env.items())
    except (AttributeError, TypeError, ValueError) as error:
        raise ExternalCommandError(
            executable,
            None,
            log_path,
            f"invalid environment mapping: {error}",
        ) from error
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in items
    ):
        raise ExternalCommandError(
            executable,
            None,
            log_path,
            "environment keys and values must all be strings",
        )
    return dict(items)
