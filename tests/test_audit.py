from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct

import nibabel as nib
import numpy as np
import pytest

from dmri_pipeline.audit import InputAuditError, audit_inputs, write_input_audit
from dmri_pipeline.utils import normalize_bvecs, round_shells, sha256_file


def _unit_directions(count: int) -> np.ndarray:
    """Return deterministic patient-free unit vectors."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    phi = golden_angle * indices
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def test_audit_accepts_matching_3_by_n_gradients(subject_config):
    audit = audit_inputs(subject_config)
    assert audit.pa_shape == (8, 8, 8, 8)
    assert audit.ap_b0_count == 2
    assert audit.shell_counts == {0: 1, 200: 1, 500: 1, 1000: 2, 2000: 1, 3000: 2}


def test_audit_accepts_unit_length_b0_vector(subject_config):
    bvecs = np.loadtxt(subject_config.bvecs)
    bvecs[:, 0] = (1.0, 0.0, 0.0)
    np.savetxt(subject_config.bvecs, bvecs, fmt="%.8f")

    assert audit_inputs(subject_config).b0_indices == (0,)


def test_audit_accepts_mixed_zero_and_unit_b0_vectors(subject_config):
    bvals = np.loadtxt(subject_config.bvals).reshape(-1)
    bvals[1] = 0.0
    np.savetxt(subject_config.bvals, bvals[None, :], fmt="%.0f")
    bvecs = np.loadtxt(subject_config.bvecs)
    bvecs[:, 0] = 0.0
    bvecs[:, 1] = (1.0, 0.0, 0.0)
    np.savetxt(subject_config.bvecs, bvecs, fmt="%.8f")

    assert audit_inputs(subject_config).b0_indices == (0, 1)


def test_audit_accepts_synthetic_177_volumes_with_11_unit_b0s_without_rewriting(
    subject_config,
):
    volume_count = 177
    b0_count = 11
    image = nib.load(subject_config.dwi_pa)
    data = np.arange(8 * 8 * 8 * volume_count, dtype=np.float32).reshape(
        8, 8, 8, volume_count
    )
    nib.save(nib.Nifti1Image(data, image.affine), subject_config.dwi_pa)
    bvals = np.concatenate(
        (
            np.zeros(b0_count),
            np.full(56, 1000.0),
            np.full(55, 2000.0),
            np.full(55, 3000.0),
        )
    )
    bvecs = np.empty((3, volume_count), dtype=float)
    bvecs[:, :b0_count] = np.array([[1.0], [0.0], [0.0]])
    bvecs[:, b0_count:] = _unit_directions(volume_count - b0_count).T
    np.savetxt(subject_config.bvals, bvals[None, :], fmt="%.0f")
    np.savetxt(subject_config.bvecs, bvecs, fmt="%.12f")
    raw_paths = (
        subject_config.dwi_pa,
        subject_config.bvals,
        subject_config.bvecs,
        subject_config.b0_ap,
    )
    before = {path: path.read_bytes() for path in raw_paths}

    result = audit_inputs(subject_config)

    assert result.pa_shape == (8, 8, 8, 177)
    assert len(result.b0_indices) == 11
    assert all(path.read_bytes() == content for path, content in before.items())


def test_audit_rejects_affine_mismatch(subject_config, rewrite_ap_affine):
    rewrite_ap_affine(subject_config.b0_ap)
    with pytest.raises(InputAuditError, match="same image grid"):
        audit_inputs(subject_config)


def test_audit_rejects_nonunit_diffusion_vectors(subject_config, corrupt_bvecs):
    corrupt_bvecs(subject_config.bvecs, norm=0.4)
    with pytest.raises(InputAuditError, match="unit length"):
        audit_inputs(subject_config)


def test_normalize_bvecs_converts_n_by_3_to_3_by_n():
    bvecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    assert np.array_equal(normalize_bvecs(bvecs, 2), bvecs.T)


@pytest.mark.parametrize(
    ("bvals", "expected"),
    [([0, 49, 50, 149, 150, 1049], [0, 0, 100, 100, 200, 1000])],
)
def test_round_shells_uses_b0_threshold_and_nearest_hundred(bvals, expected):
    assert np.array_equal(round_shells(np.array(bvals, dtype=float)), np.array(expected))


def test_normalize_bvecs_rejects_wrong_volume_count():
    with pytest.raises(InputAuditError, match="b-vector count"):
        normalize_bvecs(np.ones((3, 3)), 8)


@pytest.mark.parametrize(
    ("filename", "values"),
    [("pa_dwi.bval", "0 nan 500 1000 1000 2000 3000 3000"), ("pa_dwi.bvec", "0 1 0 0 1 1 inf 1\n0 0 1 0 1 -1 1 -1\n0 0 0 1 1 1 -1 -1")],
)
def test_audit_rejects_nonfinite_gradients(subject_config, filename, values):
    (subject_config.bvals.parent / filename).write_text(values, encoding="utf-8")
    with pytest.raises(InputAuditError, match="finite"):
        audit_inputs(subject_config)


def test_audit_rejects_wrong_bvalue_count(subject_config):
    subject_config.bvals.write_text("0 200 500", encoding="utf-8")
    with pytest.raises(InputAuditError, match="b-value count"):
        audit_inputs(subject_config)


def test_audit_rejects_intermediate_b0_vector(subject_config):
    bvecs = np.loadtxt(subject_config.bvecs)
    bvecs[:, 0] = (0.5, 0.0, 0.0)
    np.savetxt(subject_config.bvecs, bvecs)
    with pytest.raises(InputAuditError, match="b0 vectors"):
        audit_inputs(subject_config)


@pytest.mark.parametrize("norm", [0.0, 0.5])
def test_audit_rejects_zero_or_intermediate_non_b0_vector(subject_config, norm):
    bvecs = np.loadtxt(subject_config.bvecs)
    bvecs[:, 1] = (norm, 0.0, 0.0)
    np.savetxt(subject_config.bvecs, bvecs)
    with pytest.raises(InputAuditError, match="non-b0.*unit length"):
        audit_inputs(subject_config)


def test_audit_accepts_3d_ap_as_one_b0(subject_config):
    image = nib.load(subject_config.b0_ap)
    nib.save(nib.Nifti1Image(image.get_fdata()[..., 0], image.affine), subject_config.b0_ap)
    assert audit_inputs(subject_config).ap_b0_count == 1


def test_audit_rejects_non_4d_pa(subject_config):
    image = nib.load(subject_config.dwi_pa)
    nib.save(nib.Nifti1Image(image.get_fdata()[..., 0], image.affine), subject_config.dwi_pa)
    with pytest.raises(InputAuditError, match="PA image must be 4D"):
        audit_inputs(subject_config)


def test_audit_normalizes_malformed_nifti_header_errors(subject_config):
    with gzip.open(subject_config.dwi_pa, "rb") as handle:
        header = bytearray(handle.read())
    struct.pack_into("<h", header, 70, 0)
    with gzip.open(subject_config.dwi_pa, "wb") as handle:
        handle.write(header)

    with pytest.raises(InputAuditError, match="Cannot read PA image"):
        audit_inputs(subject_config)


def test_audit_rejects_invalid_ap_dimensions(subject_config):
    image = nib.load(subject_config.b0_ap)
    nib.save(nib.Nifti1Image(image.get_fdata()[..., :1, None], image.affine), subject_config.b0_ap)
    with pytest.raises(InputAuditError, match="AP image must be 3D or 4D"):
        audit_inputs(subject_config)


def test_audit_rejects_missing_b0(subject_config):
    np.savetxt(subject_config.bvals, np.full((1, 8), 200.0))
    with pytest.raises(InputAuditError, match="at least one b0"):
        audit_inputs(subject_config)


def test_audit_rejects_too_few_noncollinear_axes(subject_config):
    bvecs = np.loadtxt(subject_config.bvecs)
    bvecs[:, 1:] = np.array([[1.0], [0.0], [0.0]])
    np.savetxt(subject_config.bvecs, bvecs)
    with pytest.raises(InputAuditError, match="six unique non-collinear"):
        audit_inputs(subject_config)


def test_audit_rejects_shape_mismatch(subject_config):
    image = nib.load(subject_config.b0_ap)
    nib.save(nib.Nifti1Image(np.zeros((7, 8, 8, 2)), image.affine), subject_config.b0_ap)
    with pytest.raises(InputAuditError, match="spatial shape"):
        audit_inputs(subject_config)


def test_audit_rejects_affine_difference_beyond_tolerance(subject_config):
    image = nib.load(subject_config.b0_ap)
    affine = image.affine.copy()
    affine[0, 3] = 0.00002
    nib.save(nib.Nifti1Image(image.get_fdata(), affine), subject_config.b0_ap)
    with pytest.raises(InputAuditError, match="same image grid"):
        audit_inputs(subject_config)


def test_write_input_audit_is_deterministic_json_without_input_paths(subject_config, tmp_path):
    output = tmp_path / "audit.json"
    write_input_audit(audit_inputs(subject_config), output)
    first = output.read_text(encoding="utf-8")
    write_input_audit(audit_inputs(subject_config), output)
    assert output.read_text(encoding="utf-8") == first
    assert str(subject_config.dwi_pa) not in first
    assert json.loads(first)["pa_shape"] == [8, 8, 8, 8]


def test_sha256_file_streams_known_bytes(tmp_path, monkeypatch):
    path = tmp_path / "bytes.bin"
    payload = b"synthetic data" * 200_000
    path.write_bytes(payload)
    real_open = Path.open
    reads: list[int] = []

    class RecordingFile:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            reads.append(size)
            return self.handle.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

    def recording_open(self, *args, **kwargs):
        return RecordingFile(real_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", recording_open)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert len(reads) > 2
    assert all(0 < size <= 1024 * 1024 for size in reads)
