"""Configuration loading and validation for the dMRI pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


class ConfigError(ValueError):
    """Raised when a pipeline configuration does not meet the contract."""


@dataclass(frozen=True)
class AcquisitionConfig:
    pa_vector: tuple[int, int, int]
    ap_vector: tuple[int, int, int]
    total_readout_time: float
    slice_axis: int = 2


@dataclass(frozen=True)
class AnalysisConfig:
    dti_max_b: float = 1200.0
    noddi_workers: int | Literal["auto"] = "auto"
    ambiguous_qc_reviewed: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    subject_id: str
    dwi_pa: Path
    bvals: Path
    bvecs: Path
    b0_ap: Path
    output_root: Path
    acquisition: AcquisitionConfig
    analysis: AnalysisConfig
    fsldir: Path | None
    matlab_executable: Path | None
    config_path: Path

    @property
    def subject_output(self) -> Path:
        """Return the subject-specific output directory without creating it."""
        return self.output_root / self.subject_id

    def canonical_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable representation of this config."""
        return {
            "subject_id": self.subject_id,
            "dwi_pa": str(self.dwi_pa),
            "bvals": str(self.bvals),
            "bvecs": str(self.bvecs),
            "b0_ap": str(self.b0_ap),
            "output_root": str(self.output_root),
            "acquisition": {
                "pa_vector": list(self.acquisition.pa_vector),
                "ap_vector": list(self.acquisition.ap_vector),
                "total_readout_time": self.acquisition.total_readout_time,
                "slice_axis": self.acquisition.slice_axis,
            },
            "analysis": asdict(self.analysis),
            "fsldir": str(self.fsldir) if self.fsldir is not None else None,
            "matlab_executable": (
                str(self.matlab_executable)
                if self.matlab_executable is not None
                else None
            ),
            "config_path": str(self.config_path),
        }


_SUBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CARDINAL_AXES = {
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
}
_SCALAR_TYPES = (str, int, float, bool, type(None))


def load_config(path: Path) -> PipelineConfig:
    """Load a subject YAML file, resolving all paths relative to that file."""
    config_path = Path(path).expanduser().resolve(strict=False)
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"Cannot read configuration file: {config_path}") from error

    try:
        import yaml
    except ImportError as error:  # pragma: no cover - dependency metadata covers this
        raise ConfigError("PyYAML is required to load pipeline configuration files") from error

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}") from error

    _ensure_supported_yaml_types(data)
    root = _mapping(data, "configuration")
    base_dir = config_path.parent

    subject_id = _subject_id(_required(root, "subject_id", "configuration"))
    inputs = _mapping(_required(root, "inputs", "configuration"), "inputs")
    dwi_pa = _path(_required(inputs, "dwi_pa", "inputs"), base_dir, "inputs.dwi_pa")
    bvals = _path(_required(inputs, "bvals", "inputs"), base_dir, "inputs.bvals")
    bvecs = _path(_required(inputs, "bvecs", "inputs"), base_dir, "inputs.bvecs")
    b0_ap = _path(_required(inputs, "b0_ap", "inputs"), base_dir, "inputs.b0_ap")
    output_root = _path(
        _required(root, "output_root", "configuration"), base_dir, "output_root"
    )
    acquisition = _acquisition(_required(root, "acquisition", "configuration"))
    analysis = _analysis(root.get("analysis", {}))
    tools = _mapping(root.get("tools", {}), "tools")

    return PipelineConfig(
        subject_id=subject_id,
        dwi_pa=dwi_pa,
        bvals=bvals,
        bvecs=bvecs,
        b0_ap=b0_ap,
        output_root=output_root,
        acquisition=acquisition,
        analysis=analysis,
        fsldir=_optional_path(tools.get("fsldir"), base_dir, "tools.fsldir"),
        matlab_executable=_optional_path(
            tools.get("matlab_executable"), base_dir, "tools.matlab_executable"
        ),
        config_path=config_path,
    )


def _ensure_supported_yaml_types(value: Any) -> None:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ConfigError("YAML mapping keys must be strings")
        for child in value.values():
            _ensure_supported_yaml_types(child)
    elif isinstance(value, list):
        for child in value:
            _ensure_supported_yaml_types(child)
    elif not isinstance(value, _SCALAR_TYPES):
        raise ConfigError(f"Unsupported YAML value type: {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a YAML mapping")
    return value


def _required(mapping: dict[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key: {name}.{key}")
    return mapping[key]


def _subject_id(value: Any) -> str:
    if not isinstance(value, str) or not _SUBJECT_ID.fullmatch(value):
        raise ConfigError("subject_id must be a safe, non-empty identifier")
    return value


def _path(value: Any, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base_dir / candidate).resolve(
        strict=False
    )


def _optional_path(value: Any, base_dir: Path, name: str) -> Path | None:
    if value is None:
        return None
    return _path(value, base_dir, name)


def _acquisition(value: Any) -> AcquisitionConfig:
    mapping = _mapping(value, "acquisition")
    pa_vector = _vector(_required(mapping, "pa_vector", "acquisition"), "pa_vector")
    ap_vector = _vector(_required(mapping, "ap_vector", "acquisition"), "ap_vector")
    if pa_vector == ap_vector:
        raise ConfigError("PA and AP phase-encoding vectors must not be identical")
    readout = _positive_float(
        _required(mapping, "total_readout_time", "acquisition"), "total_readout_time"
    )
    slice_axis = mapping.get("slice_axis", 2)
    if isinstance(slice_axis, bool) or not isinstance(slice_axis, int) or slice_axis not in (0, 1, 2):
        raise ConfigError("slice_axis must be 0, 1, or 2")
    return AcquisitionConfig(pa_vector, ap_vector, readout, slice_axis)


def _vector(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{name} must be a three-element signed cardinal axis")
    if any(isinstance(component, bool) or not isinstance(component, int) for component in value):
        raise ConfigError(f"{name} must contain integer components")
    vector = tuple(value)
    if vector not in _CARDINAL_AXES:
        raise ConfigError(f"{name} must be a signed cardinal axis")
    return vector  # type: ignore[return-value]


def _analysis(value: Any) -> AnalysisConfig:
    mapping = _mapping(value, "analysis")
    dti_max_b = _positive_float(mapping.get("dti_max_b", 1200.0), "dti_max_b")
    workers = mapping.get("noddi_workers", "auto")
    if workers != "auto":
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigError("noddi_workers must be 'auto' or an integer of at least 1")
    reviewed = mapping.get("ambiguous_qc_reviewed", False)
    if not isinstance(reviewed, bool):
        raise ConfigError("ambiguous_qc_reviewed must be a boolean")
    return AnalysisConfig(dti_max_b, workers, reviewed)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a positive finite number")
    return number
