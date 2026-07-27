from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


_BVALS = np.array([0, 200, 500, 1000, 1000, 2000, 3000, 3000], dtype=float)
_BVECS = np.array(
    [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0],
    ],
    dtype=float,
)
_BVECS[:, 4:] /= np.sqrt(3.0)


@pytest.fixture
def subject_config(tmp_path: Path):
    """Build deterministic, patient-free dMRI inputs in a test directory."""
    from dmri_pipeline.config import load_config

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(
        nib.Nifti1Image(np.arange(512 * 8, dtype=np.float32).reshape(8, 8, 8, 8), affine),
        inputs / "pa_dwi.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.arange(512 * 2, dtype=np.float32).reshape(8, 8, 8, 2), affine),
        inputs / "ap_b0.nii.gz",
    )
    np.savetxt(inputs / "pa_dwi.bval", _BVALS[None, :], fmt="%.0f")
    np.savetxt(inputs / "pa_dwi.bvec", _BVECS, fmt="%.8f")
    config_path = tmp_path / "subject.yaml"
    config_path.write_text(
        """subject_id: SYNTH001
inputs:
  dwi_pa: inputs/pa_dwi.nii.gz
  bvals: inputs/pa_dwi.bval
  bvecs: inputs/pa_dwi.bvec
  b0_ap: inputs/ap_b0.nii.gz
output_root: outputs
acquisition:
  pa_vector: [0, -1, 0]
  ap_vector: [0, 1, 0]
  total_readout_time: 0.08
  slice_axis: 2
""",
        encoding="utf-8",
    )
    return load_config(config_path)


@pytest.fixture
def rewrite_ap_affine():
    def rewrite(path: Path) -> None:
        image = nib.load(path)
        affine = image.affine.copy()
        affine[0, 3] = 0.001
        nib.save(nib.Nifti1Image(image.get_fdata(), affine), path)

    return rewrite


@pytest.fixture
def corrupt_bvecs():
    def corrupt(path: Path, *, norm: float) -> None:
        bvecs = np.loadtxt(path)
        bvecs[:, 1] = (norm, 0.0, 0.0)
        np.savetxt(path, bvecs, fmt="%.8f")

    return corrupt
