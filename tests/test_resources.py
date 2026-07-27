from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dmri_pipeline.resources import ResourceValidationError, validate_jhu_resource


IMAGE_SHA256 = "974a0fd72d1214a29e58ccf33cf5aec989d937d999ae65f389dd6b3e1ffdbbad"
XML_SHA256 = "2d76ce80d1b0a50dccda2698d5eec55c8984a7f1bb438f79111d67a26fc4dc1c"
PACKAGE_ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def resource_paths() -> tuple[Path, Path]:
    root = PACKAGE_ROOT / "resources" / "jhu_48roi"
    return root / "JHU-ICBM-labels-2mm.nii.gz", root / "JHU-labels.xml"


def test_archived_jhu_resource_has_exact_labels_1_to_48(
    resource_paths: tuple[Path, Path],
) -> None:
    validation = validate_jhu_resource(*resource_paths)
    assert validation.nonzero_labels == tuple(range(1, 49))
    assert validation.image_sha256 == IMAGE_SHA256
    assert validation.xml_sha256 == XML_SHA256


def test_true_historical_names_have_uncinate_at_45_and_46_only(
    resource_paths: tuple[Path, Path],
) -> None:
    names = validate_jhu_resource(*resource_paths).label_names
    assert names[31].startswith("Sagittal stratum")
    assert "inferior fronto-occipital fasciculus" in names[31]
    assert names[32].startswith("Sagittal stratum")
    assert "inferior fronto-occipital fasciculus" in names[32]
    assert names[45] == "Uncinate fasciculus R"
    assert names[46] == "Uncinate fasciculus L"
    assert [
        label_id
        for label_id, name in names.items()
        if "inferior fronto-occipital fasciculus" in name
    ] == [31, 32]
    with pytest.raises(TypeError):
        names[45] = "changed"  # type: ignore[index]


def test_provenance_has_complete_true_mapping_and_exact_resource_identity(
    resource_paths: tuple[Path, Path],
) -> None:
    image_path, xml_path = resource_paths
    provenance = json.loads(
        (image_path.parent / "provenance.json").read_text(encoding="utf-8")
    )
    validation = validate_jhu_resource(image_path, xml_path)
    assert provenance["source"] == {
        "component": "data_atlases",
        "tag": "fsl-5_0_4",
    }
    assert provenance["files"] == {
        "JHU-ICBM-labels-2mm.nii.gz": {"sha256": _sha256(image_path)},
        "JHU-labels.xml": {"sha256": _sha256(xml_path)},
    }
    assert provenance["nonzero_labels"] == list(range(1, 49))
    assert provenance["nearest_neighbour_only"] is True
    assert provenance["label_mapping"] == {
        str(label_id): validation.label_names[label_id] for label_id in range(49)
    }


def _copy_resources(
    tmp_path: Path, resource_paths: tuple[Path, Path]
) -> tuple[Path, Path]:
    image_path = tmp_path / "atlas.nii.gz"
    xml_path = tmp_path / "atlas.xml"
    shutil.copyfile(resource_paths[0], image_path)
    shutil.copyfile(resource_paths[1], xml_path)
    return image_path, xml_path


def _rewrite_image(path: Path, transform) -> None:
    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=np.float64)
    transformed = transform(data.copy())
    nib.save(nib.Nifti1Image(transformed, image.affine), path)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda data: np.where(data == 48, 47, data), "exactly 1 through 48"),
        (lambda data: np.where(data == 0, 49, data), "exactly 1 through 48"),
        (lambda data: np.where(data == 1, -1, data), "exactly 1 through 48"),
        (lambda data: np.where(data == 1, 1.5, data), "integer-valued"),
        (lambda data: np.where(data == 1, np.nan, data), "finite"),
    ],
)
def test_atlas_rejects_missing_extra_fractional_and_nonfinite_labels(
    tmp_path: Path,
    resource_paths: tuple[Path, Path],
    transform,
    message: str,
) -> None:
    image_path, xml_path = _copy_resources(tmp_path, resource_paths)
    _rewrite_image(image_path, transform)
    with pytest.raises(ResourceValidationError, match=message):
        validate_jhu_resource(image_path, xml_path)


def test_valid_but_noncanonical_atlas_bytes_are_rejected(
    tmp_path: Path, resource_paths: tuple[Path, Path]
) -> None:
    image_path, xml_path = _copy_resources(tmp_path, resource_paths)
    image = nib.load(image_path)
    nib.save(
        nib.Nifti1Image(np.asarray(image.dataobj), image.affine, image.header),
        image_path,
    )
    with pytest.raises(ResourceValidationError, match="SHA-256"):
        validate_jhu_resource(image_path, xml_path)


def test_malformed_nifti_is_normalized_to_resource_error(
    tmp_path: Path, resource_paths: tuple[Path, Path]
) -> None:
    image_path, xml_path = _copy_resources(tmp_path, resource_paths)
    image_path.write_bytes(b"not a nifti")
    with pytest.raises(ResourceValidationError, match="Cannot read"):
        validate_jhu_resource(image_path, xml_path)


def test_nonfinite_atlas_affine_is_rejected(
    tmp_path: Path, resource_paths: tuple[Path, Path]
) -> None:
    image_path, xml_path = _copy_resources(tmp_path, resource_paths)
    image = nib.load(image_path)
    header = image.header.copy()
    header["sform_code"] = 1
    header["srow_x"][0] = np.nan
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), None, header), image_path)
    with pytest.raises(ResourceValidationError, match="affine.*finite"):
        validate_jhu_resource(image_path, xml_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('index="48"', 'index="47"', "duplicate XML index"),
        (
            '<label index="48" x="117" y="76" z="87">Tapetum L</label>',
            "",
            "indices exactly 0 through 48",
        ),
        (">Tapetum L</label>", "> </label>", "nonempty"),
        (
            ">Tapetum L</label>",
            ">Tapetum R</label>",
            "unique",
        ),
        ('index="48"', 'index="48.0"', "index must be an integer"),
    ],
)
def test_xml_rejects_duplicate_missing_empty_and_duplicate_names(
    tmp_path: Path,
    resource_paths: tuple[Path, Path],
    old: str,
    new: str,
    message: str,
) -> None:
    image_path, xml_path = _copy_resources(tmp_path, resource_paths)
    xml_path.write_text(
        xml_path.read_text(encoding="iso-8859-1").replace(old, new, 1),
        encoding="iso-8859-1",
    )
    with pytest.raises(ResourceValidationError, match=message):
        validate_jhu_resource(image_path, xml_path)


def test_valid_but_noncanonical_xml_bytes_are_rejected(
    tmp_path: Path, resource_paths: tuple[Path, Path]
) -> None:
    image_path, xml_path = _copy_resources(tmp_path, resource_paths)
    xml_path.write_bytes(xml_path.read_bytes() + b"\n")
    with pytest.raises(ResourceValidationError, match="XML SHA-256"):
        validate_jhu_resource(image_path, xml_path)
