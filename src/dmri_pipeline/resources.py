"""Identity and label-table validation for the archived 48-label JHU atlas."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError, ImageDataError


JHU_IMAGE_SHA256 = "974a0fd72d1214a29e58ccf33cf5aec989d937d999ae65f389dd6b3e1ffdbbad"
JHU_XML_SHA256 = "2d76ce80d1b0a50dccda2698d5eec55c8984a7f1bb438f79111d67a26fc4dc1c"
_EXPECTED_INDICES = tuple(range(49))
_EXPECTED_NONZERO = tuple(range(1, 49))


class ResourceValidationError(ValueError):
    """Raised when the packaged JHU atlas is not the accepted 48-label resource."""


@dataclass(frozen=True)
class AtlasValidation:
    """Immutable identity and true-value label mapping for the JHU resource."""

    image_shape: tuple[int, int, int]
    image_sha256: str
    xml_sha256: str
    nonzero_labels: tuple[int, ...]
    label_names: Mapping[int, str]


def validate_jhu_resource(image_path: Path, xml_path: Path) -> AtlasValidation:
    """Validate exact resource bytes, atlas values, and true XML label indices."""
    image_file = Path(image_path)
    xml_file = Path(xml_path)
    data, shape = _read_atlas(image_file)
    _validate_atlas_values(data)
    names = _parse_label_names(xml_file)

    image_hash = _sha256(image_file, "atlas image")
    if image_hash != JHU_IMAGE_SHA256:
        raise ResourceValidationError(
            "JHU atlas image SHA-256 does not match the accepted FSL 5.0.4 resource"
        )
    xml_hash = _sha256(xml_file, "atlas XML")
    if xml_hash != JHU_XML_SHA256:
        raise ResourceValidationError(
            "JHU atlas XML SHA-256 does not match the accepted FSL 5.0.4 resource"
        )

    return AtlasValidation(
        image_shape=shape,
        image_sha256=image_hash,
        xml_sha256=xml_hash,
        nonzero_labels=_EXPECTED_NONZERO,
        label_names=MappingProxyType(dict(names)),
    )


def _read_atlas(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    try:
        image = nib.load(path)
        data = np.asarray(image.dataobj, dtype=np.float64)
    except (
        OSError,
        ValueError,
        ImageFileError,
        HeaderDataError,
        ImageDataError,
    ) as error:
        raise ResourceValidationError("Cannot read JHU atlas image") from error
    if len(image.shape) != 3:
        raise ResourceValidationError("JHU atlas image must be 3D")
    if not np.issubdtype(image.get_data_dtype(), np.number) or np.issubdtype(
        image.get_data_dtype(), np.complexfloating
    ):
        raise ResourceValidationError("JHU atlas image must be numeric")
    if not np.isfinite(image.affine).all():
        raise ResourceValidationError("JHU atlas image affine must be finite")
    return data, tuple(int(value) for value in image.shape)


def _validate_atlas_values(data: np.ndarray) -> None:
    if not np.isfinite(data).all():
        raise ResourceValidationError("JHU atlas voxels must be finite")
    if not np.equal(data, np.rint(data)).all():
        raise ResourceValidationError("JHU atlas voxels must be integer-valued")
    labels = tuple(int(value) for value in np.unique(data) if value != 0)
    if labels != _EXPECTED_NONZERO:
        raise ResourceValidationError(
            "JHU atlas nonzero labels must be exactly 1 through 48"
        )
    if np.any(data < 0) or np.any(data > 48):
        raise ResourceValidationError("JHU atlas values must be between 0 and 48")


def _parse_label_names(path: Path) -> dict[int, str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError, ValueError) as error:
        raise ResourceValidationError("Cannot parse JHU atlas XML") from error

    names: dict[int, str] = {}
    for element in root.findall(".//label"):
        raw_index = element.get("index")
        try:
            index = int(raw_index) if raw_index is not None else None
        except ValueError as error:
            raise ResourceValidationError("JHU XML label index must be an integer") from error
        if index is None or str(index) != raw_index:
            raise ResourceValidationError("JHU XML label index must be an integer")
        if index in names:
            raise ResourceValidationError(f"JHU XML has duplicate XML index {index}")
        name = "".join(element.itertext()).strip()
        if not name:
            raise ResourceValidationError("JHU XML label names must be nonempty")
        names[index] = name

    if tuple(sorted(names)) != _EXPECTED_INDICES:
        raise ResourceValidationError("JHU XML must contain indices exactly 0 through 48")
    if len(set(names.values())) != len(names):
        raise ResourceValidationError("JHU XML label names must be unique")
    if names[45] != "Uncinate fasciculus R":
        raise ResourceValidationError("JHU XML label 45 must be Uncinate fasciculus R")
    if names[46] != "Uncinate fasciculus L":
        raise ResourceValidationError("JHU XML label 46 must be Uncinate fasciculus L")
    if "Sagittal stratum" not in names[31] or "Sagittal stratum" not in names[32]:
        raise ResourceValidationError("JHU XML labels 31 and 32 must be Sagittal stratum")
    if any(
        "inferior fronto-occipital fasciculus" in name and index not in (31, 32)
        for index, name in names.items()
    ):
        raise ResourceValidationError("JHU XML must not contain separate IFOF labels")
    return names


def _sha256(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ResourceValidationError(f"Cannot hash JHU {label}") from error
    return digest.hexdigest()
