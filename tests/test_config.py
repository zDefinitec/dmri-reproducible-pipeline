from __future__ import annotations

from pathlib import Path

import pytest

from dmri_pipeline.config import ConfigError, load_config


def write_subject_yaml(
    directory: Path,
    *,
    subject_id: str = "SUB001",
    pa_vector: list[int] | None = None,
    ap_vector: list[int] | None = None,
    total_readout_time: float = 0.08,
    extra: str = "",
) -> Path:
    pa_vector = pa_vector or [0, -1, 0]
    ap_vector = ap_vector or [0, 1, 0]
    path = directory / "subject.yaml"
    path.write_text(
        "\n".join(
            [
                f"subject_id: {subject_id!r}",
                "inputs:",
                "  dwi_pa: inputs/pa_dwi.nii.gz",
                "  bvals: inputs/pa_dwi.bval",
                "  bvecs: inputs/pa_dwi.bvec",
                "  b0_ap: inputs/ap_b0.nii.gz",
                "output_root: outputs",
                "acquisition:",
                f"  pa_vector: {pa_vector}",
                f"  ap_vector: {ap_vector}",
                f"  total_readout_time: {total_readout_time}",
                "slice_axis: 2",
                extra.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_paths_and_normalizes_vectors(tmp_path: Path) -> None:
    config_path = write_subject_yaml(tmp_path)

    config = load_config(config_path)

    assert config.subject_id == "SUB001"
    assert config.acquisition.pa_vector == (0, -1, 0)
    assert config.acquisition.total_readout_time == pytest.approx(0.08)
    assert config.subject_output == tmp_path / "outputs" / "SUB001"
    assert config.dwi_pa == tmp_path / "inputs" / "pa_dwi.nii.gz"


@pytest.mark.parametrize("subject_id", ["../escape", "A/B", "", "subject name"])
def test_subject_id_rejects_unsafe_values(tmp_path: Path, subject_id: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write_subject_yaml(tmp_path, subject_id=subject_id))


@pytest.mark.parametrize(
    "extra",
    [
        "acquisition:\n  pa_vector: [1, 1, 0]\n  ap_vector: [0, 1, 0]\n  total_readout_time: 0.1",
        "acquisition:\n  pa_vector: [0, -1, 0]\n  ap_vector: [0, -1, 0]\n  total_readout_time: 0.1",
        "acquisition:\n  pa_vector: [0, -1, 0]\n  ap_vector: [0, 1, 0]\n  total_readout_time: 0",
        "acquisition:\n  pa_vector: [0, -1, 0]\n  ap_vector: [0, 1, 0]\n  total_readout_time: .nan",
    ],
)
def test_acquisition_rejects_invalid_vectors_and_readout(
    tmp_path: Path, extra: str
) -> None:
    with pytest.raises(ConfigError):
        load_config(write_subject_yaml(tmp_path, extra=extra))


def test_optional_analysis_validation_and_canonical_dict(tmp_path: Path) -> None:
    config = load_config(
        write_subject_yaml(
            tmp_path,
            extra="""analysis:
  dti_max_b: 1000
  noddi_workers: auto
  ambiguous_qc_reviewed: true
tools:
  fsldir: toolchain/fsl
  matlab_executable: toolchain/MATLAB
""",
        )
    )

    assert config.canonical_dict() == {
        "subject_id": "SUB001",
        "dwi_pa": str(tmp_path / "inputs" / "pa_dwi.nii.gz"),
        "bvals": str(tmp_path / "inputs" / "pa_dwi.bval"),
        "bvecs": str(tmp_path / "inputs" / "pa_dwi.bvec"),
        "b0_ap": str(tmp_path / "inputs" / "ap_b0.nii.gz"),
        "output_root": str(tmp_path / "outputs"),
        "acquisition": {
            "pa_vector": [0, -1, 0],
            "ap_vector": [0, 1, 0],
            "total_readout_time": 0.08,
            "slice_axis": 2,
        },
        "analysis": {
            "dti_max_b": 1000.0,
            "noddi_workers": "auto",
            "ambiguous_qc_reviewed": True,
        },
        "fsldir": str(tmp_path / "toolchain" / "fsl"),
        "matlab_executable": str(tmp_path / "toolchain" / "MATLAB"),
        "config_path": str(tmp_path / "subject.yaml"),
    }


@pytest.mark.parametrize(
    "extra",
    [
        "acquisition:\n  pa_vector: [0, -1, 0]\n  ap_vector: [0, 1, 0]\n  total_readout_time: 0.1\n  slice_axis: 3",
        "analysis:\n  noddi_workers: 0",
        "analysis:\n  dti_max_b: .inf",
        "analysis:\n  noddi_workers: many",
    ],
)
def test_analysis_and_slice_axis_reject_invalid_values(tmp_path: Path, extra: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write_subject_yaml(tmp_path, extra=extra))
