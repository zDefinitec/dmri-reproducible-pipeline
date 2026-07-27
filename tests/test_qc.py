from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pytest

from dmri_pipeline.qc import (
    FIGURE_FILENAMES,
    STAGE_FIGURES,
    QCError,
    StageQCContext,
    generate_all_qc,
    generate_stage_qc,
)
from dmri_pipeline.state import StageContext
from dmri_pipeline.utils import round_shells


@dataclass(frozen=True)
class SyntheticQCCase:
    context: StageQCContext
    paths: dict[str, Path]


def make_qc_case(subject_config, *, ap_3d: bool = False) -> SyntheticQCCase:
    subject_root = subject_config.subject_output
    subject_root.mkdir(parents=True)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shape = (8, 8, 8)
    bvals = np.loadtxt(subject_config.bvals).reshape(-1)
    raw_pa = nib.load(subject_config.dwi_pa)
    raw_pa_data = np.asarray(raw_pa.dataobj, dtype=np.float32)
    raw_ap_data = np.asarray(nib.load(subject_config.b0_ap).dataobj, dtype=np.float32)
    if ap_3d:
        raw_ap_data = raw_ap_data[..., 0]
        nib.save(nib.Nifti1Image(raw_ap_data, affine), subject_config.b0_ap)
    ap_count = 1 if raw_ap_data.ndim == 3 else raw_ap_data.shape[3]

    paths: dict[str, Path] = {}

    def image(name: str, values: np.ndarray) -> Path:
        path = subject_root / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine), path)
        paths[name] = path
        return path

    def document(name: str, text: str) -> Path:
        path = subject_root / name
        path.write_text(text, encoding="utf-8")
        paths[name] = path
        return path

    ap_4d = raw_ap_data[..., None] if raw_ap_data.ndim == 3 else raw_ap_data
    denoised_pa = image("denoised_pa", raw_pa_data * 0.99 + 0.1)
    denoised_ap = image("denoised_ap", ap_4d * 0.99 + 0.1)
    gibbs_pa = image("gibbs_pa", raw_pa_data * 0.98 + 0.2)
    gibbs_ap = image("gibbs_ap", ap_4d * 0.98 + 0.2)
    pa_indices = np.flatnonzero(bvals < 50)
    merged = np.concatenate(
        (raw_pa_data[..., pa_indices], ap_4d), axis=3
    )
    topup_merged = image("topup_merged", merged)
    topup_corrected = image("topup_corrected", merged * 0.97 + 0.3)
    hifi = image("hifi", np.mean(merged * 0.97 + 0.3, axis=3))
    brain_values = np.zeros(shape, dtype=np.uint8)
    brain_values[1:7, 1:7, 1:7] = 1
    brain_mask = image("brain_mask", brain_values)
    eddy_dwi = image("eddy_dwi", raw_pa_data * 0.96 + 0.4)

    base = np.linspace(0.1, 0.9, np.prod(shape), dtype=np.float32).reshape(shape)

    def maps(prefix: str, keys: tuple[str, ...]) -> dict[str, Path]:
        return {
            key: image(f"{prefix}_{key}", base + index * 0.01)
            for index, key in enumerate(keys)
        }

    dti_maps = maps("dti", ("FA", "MD", "AD", "RD"))
    dki_maps = maps("dki", ("FA", "MD", "AD", "RD", "MK", "AK", "RK"))
    direct_maps = maps("direct", ("MD", "MK", "S0"))
    noddi_maps = maps("noddi", ("ODI", "FICVF", "FISO"))
    atlas = np.zeros(shape, dtype=np.uint8)
    for label, coordinate in enumerate(np.argwhere(brain_values == 1)[:48], 1):
        atlas[tuple(coordinate)] = label
    warped_atlas = image("warped_atlas", atlas)

    stripe_directory = subject_root / "00_pre_denoise_motion_qc"
    stripe_directory.mkdir()
    stripe_csv = stripe_directory / "stripe_metrics.csv"
    csi = np.array([1.0, 1.16, 1.31, 1.0, 1.0, 1.0, 1.0, 1.0])
    with stripe_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "volume_index_zero_based",
                "volume_number_one_based",
                "b_value",
                "nominal_shell",
                "a_si",
                "c_si",
                "classification",
                "peak_sagittal_index_zero_based",
                "peak_sagittal_number_one_based",
            )
        )
        for index, (b_value, value, shell) in enumerate(
            zip(bvals, csi, round_shells(bvals), strict=True)
        ):
            classification = (
                "high" if value > 1.25 else "ambiguous" if value >= 1.15 else "normal"
            )
            writer.writerow(
                (index, index + 1, b_value, shell, 100 + index, value,
                 classification, 3, 4)
            )
    paths["stripe_metrics.csv"] = stripe_csv
    decision = {
        "subject_id": subject_config.subject_id,
        "method": "Henrique Appendix A sagittal stripe index",
        "thresholds": {
            "ambiguous_min_inclusive": 1.15,
            "high_min_exclusive": 1.25,
            "exclude_high_volume_count": 5,
        },
        "decision": "HOLD_FOR_REVIEW",
        "exit_code": 21,
        "ambiguous_reviewed": False,
        "volume_counts": {"total": 8, "normal": 6, "ambiguous": 1, "high": 1},
        "flagged_indices_zero_based": {"high": [2], "ambiguous": [1]},
        "flagged_volume_numbers_one_based": {"high": [3], "ambiguous": [2]},
        "maximum_csi": {
            "value": 1.31,
            "volume_index_zero_based": 2,
            "volume_number_one_based": 3,
        },
        "shell_counts": {
            str(int(shell)): int(np.count_nonzero(round_shells(bvals) == shell))
            for shell in np.unique(round_shells(bvals))
        },
    }
    stripe_decision = stripe_directory / "stripe_decision.json"
    stripe_decision.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["stripe_decision.json"] = stripe_decision
    topup_manifest = document(
        "topup_manifest.json",
        json.dumps(
            {
                "pa_b0_count": len(pa_indices),
                "ap_b0_count": ap_count,
                "combined_b0_count": len(pa_indices) + ap_count,
                "volume_order": ["PA"] * len(pa_indices) + ["AP"] * ap_count,
                "eddy_acquisition_row_order": ["PA", "AP"],
            },
            indent=2,
        )
        + "\n",
    )
    parameters = np.column_stack(
        (
            np.linspace(0, 0.7, 8),
            np.linspace(0, -0.35, 8),
            np.zeros(8),
            np.linspace(0, 0.007, 8),
            np.zeros(8),
            np.zeros(8),
        )
    )
    eddy_parameters = subject_root / "eddy_parameters.txt"
    np.savetxt(eddy_parameters, parameters)
    paths["eddy_parameters.txt"] = eddy_parameters
    movement = np.column_stack((np.linspace(0, 0.4, 8), np.linspace(0, 0.2, 8)))
    eddy_rms = subject_root / "eddy_rms.txt"
    np.savetxt(eddy_rms, movement)
    paths["eddy_rms.txt"] = eddy_rms
    outlier = document(
        "eddy_outlier_map.txt",
        "Slice 0 1 2 3\n"
        + "\n".join(
            " ".join(str(value) for value in row)
            for row in np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )
        + "\n",
    )
    output = subject_root / "qc"
    output.mkdir()
    stage_context = StageContext(
        subject_config,
        Path(__file__).parents[1],
        subject_root,
        {"python": "3.11", "matplotlib": matplotlib.__version__},
    )
    context = StageQCContext(
        stage_context=stage_context,
        output_directory=output,
        bvals=subject_config.bvals,
        raw_pa=subject_config.dwi_pa,
        raw_ap=subject_config.b0_ap,
        stripe_metrics_csv=stripe_csv,
        stripe_decision_json=stripe_decision,
        denoised_pa=denoised_pa,
        denoised_ap=denoised_ap,
        gibbs_pa=gibbs_pa,
        gibbs_ap=gibbs_ap,
        topup_merged_b0=topup_merged,
        topup_corrected_b0=topup_corrected,
        topup_manifest_json=topup_manifest,
        hifi_b0=hifi,
        brain_mask=brain_mask,
        eddy_dwi=eddy_dwi,
        eddy_parameters=eddy_parameters,
        eddy_movement_rms=eddy_rms,
        eddy_outlier_map=outlier,
        dti_maps=dti_maps,
        dki_maps=dki_maps,
        dki_direct_maps=direct_maps,
        noddi_maps=noddi_maps,
        warped_atlas=warped_atlas,
    )
    return SyntheticQCCase(context, paths)


@pytest.fixture
def completed_context(subject_config) -> SyntheticQCCase:
    return make_qc_case(subject_config)


def test_qc_manifest_contract_is_frozen() -> None:
    assert tuple(FIGURE_FILENAMES.items()) == (
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
    assert tuple(STAGE_FIGURES) == (
        "00_input_audit",
        "00_pre_denoise_motion_qc",
        "01_denoise",
        "02_gibbs",
        "03_topup",
        "04_bet",
        "05_eddy",
        "06_dti",
        "07_dki",
        "07_dki_direct",
        "08_noddi",
        "09_jhu_48roi",
        "overview",
    )


def test_qc_manifest_contains_all_scientific_stages(
    completed_context: SyntheticQCCase,
) -> None:
    manifest = generate_all_qc(completed_context.context)
    assert tuple(manifest) == tuple(FIGURE_FILENAMES)
    assert set(manifest) >= {
        "input",
        "stripe",
        "denoise_pa",
        "denoise_ap",
        "gibbs_pa",
        "gibbs_ap",
        "topup",
        "bet",
        "eddy_images",
        "eddy_motion",
        "dti",
        "dki",
        "dki_direct",
        "noddi",
        "jhu_48roi",
        "overview",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in manifest.values())
    for path in manifest.values():
        pixels = mpimg.imread(path)
        assert min(pixels.shape[:2]) > 0
        assert np.isfinite(pixels).all()
    payload = json.loads(
        (completed_context.context.output_directory / "qc_manifest.json").read_text()
    )
    assert tuple(payload["figures"]) == tuple(FIGURE_FILENAMES)
    assert payload["visual_review_status"] == "NOT_REVIEWED"
    assert payload["subject_id"] == "SYNTH001"
    assert payload["pre_denoise_detail_directory"] == "00_pre_denoise_motion_qc"
    serialized = json.dumps(payload)
    assert "/Users/" not in serialized
    assert ".work" not in serialized
    for figure_id, path in manifest.items():
        pixels = mpimg.imread(path)
        assert payload["figure_metadata"][figure_id]["width"] == pixels.shape[1]
        assert payload["figure_metadata"][figure_id]["height"] == pixels.shape[0]


def test_stage_qc_supports_3d_ap_and_streams_without_get_fdata(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_qc_case(subject_config, ap_3d=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("QC must not call get_fdata for a 4D DWI mean")

    monkeypatch.setattr(nib.spatialimages.SpatialImage, "get_fdata", forbidden)
    paths = generate_stage_qc(case.context, "00_input_audit")
    assert [path.name for path in paths] == [
        "00_input_b0.png",
        "00_shell_distribution.png",
    ]


def test_qc_refuses_unknown_duplicate_and_nonempty_destination(
    completed_context: SyntheticQCCase,
) -> None:
    with pytest.raises(QCError, match="unknown"):
        generate_stage_qc(completed_context.context, "not-a-stage")
    generated = generate_stage_qc(completed_context.context, "04_bet")
    assert len(generated) == 1
    with pytest.raises(QCError, match="empty"):
        generate_stage_qc(completed_context.context, "04_bet")


def test_qc_rejects_malformed_48_label_atlas(subject_config) -> None:
    case = make_qc_case(subject_config)
    image = nib.load(case.context.warped_atlas)
    values = np.asarray(image.dataobj).copy()
    values[values == 48] = 47
    nib.save(nib.Nifti1Image(values, image.affine), case.context.warped_atlas)
    with pytest.raises(QCError, match="1 through 48"):
        generate_stage_qc(case.context, "09_jhu_48roi")


def test_qc_detects_input_mutation_and_rolls_back(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original = qc_module._render_shells

    def mutate(context, inputs, path):
        original(context, inputs, path)
        context.bvals.write_text(
            context.bvals.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

    monkeypatch.setattr(qc_module, "_render_shells", mutate)
    with pytest.raises(QCError, match="changed"):
        generate_stage_qc(case.context, "00_input_audit")
    assert list(case.context.output_directory.iterdir()) == []


def test_context_rejects_model_key_and_symlink_attacks(
    subject_config, tmp_path: Path
) -> None:
    case = make_qc_case(subject_config)
    with pytest.raises(QCError, match="exact keys"):
        StageQCContext(
            **{
                **case.context.__dict__,
                "dti_maps": {"FA": case.context.dti_maps["FA"]},
            }
        )
    link = tmp_path / "linked_mask.nii.gz"
    link.symlink_to(case.context.brain_mask)
    with pytest.raises(QCError, match="symbolic"):
        StageQCContext(**{**case.context.__dict__, "brain_mask": link})


def test_qc_rejects_bad_grid_mask_topup_eddy_and_outlier_inputs(
    subject_config,
) -> None:
    case = make_qc_case(subject_config)
    root = case.context.stage_context.subject_root
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    bad_mask = root / "bad_mask.nii.gz"
    values = np.ones((8, 8, 8), dtype=np.float32)
    values.flat[0] = 2
    nib.save(nib.Nifti1Image(values, affine), bad_mask)
    (root / "bad_qc1").mkdir()
    with pytest.raises(QCError, match="binary"):
        generate_stage_qc(
            replace(case.context, brain_mask=bad_mask, output_directory=root / "bad_qc1"),
            "04_bet",
        )

    bad_fa = root / "bad_fa.nii.gz"
    shifted = affine.copy()
    shifted[0, 3] = 0.01
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), shifted), bad_fa
    )
    dti = dict(case.context.dti_maps)
    dti["FA"] = bad_fa
    (root / "bad_qc2").mkdir()
    with pytest.raises(QCError, match="grid"):
        generate_stage_qc(
            replace(case.context, dti_maps=dti, output_directory=root / "bad_qc2"),
            "06_dti",
        )

    bad_manifest = root / "bad_topup.json"
    payload = json.loads(case.context.topup_manifest_json.read_text(encoding="utf-8"))
    payload["volume_order"] = ["AP", "PA", "AP"]
    bad_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (root / "bad_qc3").mkdir()
    with pytest.raises(QCError, match="PA-then-AP"):
        generate_stage_qc(
            replace(
                case.context,
                topup_manifest_json=bad_manifest,
                output_directory=root / "bad_qc3",
            ),
            "03_topup",
        )

    bad_parameters = root / "bad_parameters.txt"
    np.savetxt(bad_parameters, np.zeros((7, 6)))
    (root / "bad_qc4").mkdir()
    with pytest.raises(QCError, match="one row"):
        generate_stage_qc(
            replace(
                case.context,
                eddy_parameters=bad_parameters,
                output_directory=root / "bad_qc4",
            ),
            "05_eddy",
        )

    bad_outliers = root / "bad_outliers.txt"
    np.savetxt(bad_outliers, np.full((8, 4), 2), fmt="%d")
    (root / "bad_qc5").mkdir()
    with pytest.raises(QCError, match="0/1"):
        generate_stage_qc(
            replace(
                case.context,
                eddy_outlier_map=bad_outliers,
                output_directory=root / "bad_qc5",
            ),
            "05_eddy",
        )


def test_qc_rejects_hardlink_fifo_and_parent_traversal(
    subject_config,
) -> None:
    case = make_qc_case(subject_config)
    root = case.context.stage_context.subject_root
    hardlink = root / "hardlinked_fa.nii.gz"
    os.link(case.context.dti_maps["FA"], hardlink)
    maps = dict(case.context.dki_maps)
    maps["FA"] = hardlink
    (root / "hard_qc").mkdir()
    with pytest.raises(QCError, match="hard-link"):
        generate_stage_qc(
            replace(case.context, dki_maps=maps, output_directory=root / "hard_qc"),
            "07_dki",
        )
    fifo = root / "mask_fifo"
    os.mkfifo(fifo)
    (root / "fifo_qc").mkdir()
    with pytest.raises(QCError, match="regular"):
        generate_stage_qc(
            replace(case.context, brain_mask=fifo, output_directory=root / "fifo_qc"),
            "04_bet",
        )
    with pytest.raises(QCError, match="parent traversal"):
        replace(case.context, output_directory=root / "nested" / ".." / "qc")


def test_qc_partial_commit_failure_removes_owned_output(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    real_write = qc_module.os.write
    calls = 0

    def short_then_fail(descriptor, content):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, content[:10])
        raise OSError("injected short-write failure")

    monkeypatch.setattr(qc_module.os, "write", short_then_fail)
    with pytest.raises(QCError, match="commit"):
        generate_stage_qc(case.context, "04_bet")
    assert list(case.context.output_directory.iterdir()) == []


def test_snapshot_temp_does_not_use_pathname_temp_helpers(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pathname-based temporary helper was called")

    monkeypatch.setattr(qc_module.tempfile, "TemporaryDirectory", forbidden)
    with qc_module._InputGuard([subject_config.bvals]):
        pass


def test_render_temp_does_not_use_pathname_temp_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pathname-based temporary helper was called")

    monkeypatch.setattr(qc_module.tempfile, "mkdtemp", forbidden)
    with qc_module._private_temp(tmp_path):
        pass


def test_owned_temp_post_mkdir_stat_failure_removes_exact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    real_stat = qc_module.os.stat
    failed = False

    def fail_first_root_stat(
        path,
        *,
        dir_fd=None,
        follow_symlinks=True,
    ):
        nonlocal failed
        if (
            not failed
            and dir_fd is not None
            and str(path).startswith(".post-mkdir-")
        ):
            failed = True
            raise OSError("injected post-mkdir identity failure")
        return real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(qc_module.os, "stat", fail_first_root_stat)
    with pytest.raises(OSError, match="post-mkdir identity failure"):
        qc_module._OwnedDirectory.create(
            tmp_path,
            prefix=".post-mkdir-",
        )

    assert failed
    assert not any(
        path.name.startswith(".post-mkdir-")
        for path in tmp_path.iterdir()
    )


def test_owned_temp_post_mkdir_swap_preserves_foreign_and_removes_exact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    real_stat = qc_module.os.stat
    state: dict[str, Path] = {}

    def swap_on_first_named_stat(
        path,
        *,
        dir_fd=None,
        follow_symlinks=True,
    ):
        if (
            not state
            and dir_fd is not None
            and str(path).startswith(".post-mkdir-swap-")
        ):
            original = tmp_path / str(path)
            held = original.with_name(f"{original.name}.held")
            original.rename(held)
            original.mkdir(mode=0o700)
            state.update(original=original, held=held)
            raise OSError("injected post-mkdir binding swap")
        return real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(qc_module.os, "stat", swap_on_first_named_stat)
    try:
        with pytest.raises(OSError, match="post-mkdir binding swap"):
            qc_module._OwnedDirectory.create(
                tmp_path,
                prefix=".post-mkdir-swap-",
            )

        assert state["original"].exists()
        assert list(state["original"].iterdir()) == []
        assert not state["held"].exists()
    finally:
        for path in state.values():
            if path.exists():
                path.rmdir()


def test_owned_temp_reserve_fstat_failure_closes_fds_and_erases_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    real_open = qc_module.os.open
    real_fstat = qc_module.os.fstat
    state: dict[str, int] = {}

    def track_child_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "child.bin" and dir_fd is not None:
            state["descriptor"] = descriptor
        return descriptor

    def fail_child_fstat(descriptor):
        if descriptor == state.get("descriptor"):
            raise OSError("injected reserve fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(qc_module.os, "open", track_child_open)
    monkeypatch.setattr(qc_module.os, "fstat", fail_child_fstat)
    with pytest.raises(OSError, match="reserve fstat failure"):
        qc_module._OwnedDirectory.create(
            tmp_path,
            prefix=".reserve-",
            names=("child.bin",),
        )

    with pytest.raises(OSError):
        real_fstat(state["descriptor"])
    assert not any(
        path.name.startswith(".reserve-")
        for path in tmp_path.iterdir()
    )


def test_owned_temp_writer_fdopen_failure_closes_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    owner = qc_module._private_temp(
        tmp_path,
        names=("derived.bin",),
    )
    real_dup = qc_module.os.dup
    real_fstat = qc_module.os.fstat
    state: dict[str, int] = {}

    def track_dup(descriptor):
        duplicate = real_dup(descriptor)
        state["duplicate"] = duplicate
        return duplicate

    def fail_fdopen(*_args, **_kwargs):
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(qc_module.os, "dup", track_dup)
    monkeypatch.setattr(qc_module.os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(OSError, match="fdopen failure"):
            with owner.writer("derived.bin"):
                pass
        with pytest.raises(OSError):
            real_fstat(state["duplicate"])
    finally:
        owner.close()


def test_qc_render_temp_cleanup_preserves_reused_foreign_root(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original = qc_module._render_bet
    state: dict[str, Path] = {}
    foreign_bytes = b"foreign QC staging root\n"

    def rename_root_after_render(context, inputs, path):
        original(context, inputs, path)
        root = path.parent
        held = root.with_name(f"{root.name}.held-qc-render")
        root.rename(held)
        (root / "nested").mkdir(parents=True, mode=0o700)
        foreign = root / "nested" / "foreign.bin"
        foreign.write_bytes(foreign_bytes)
        state.update(root=root, held=held, foreign=foreign)

    monkeypatch.setattr(qc_module, "_render_bet", rename_root_after_render)
    try:
        with pytest.raises(QCError, match="temporary|staging|binding|changed"):
            generate_stage_qc(case.context, "04_bet")

        assert state["foreign"].read_bytes() == foreign_bytes
        assert not state["held"].exists() or list(state["held"].iterdir()) == []
        if state["held"].exists():
            assert not any(path.is_file() for path in state["held"].rglob("*"))
        assert list(case.context.output_directory.iterdir()) == []
    finally:
        for key in ("held", "root"):
            candidate = state.get(key)
            if candidate is not None and candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_qc_render_writes_owned_fd_not_foreign_replacement(
    subject_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original = qc_module._render_bet
    state: dict[str, Path] = {}
    foreign_bytes = b"foreign staging target must remain unchanged\n"

    def swap_before_render(context, inputs, target):
        root = target.parent
        held = root.with_name(f"{root.name}.held-before-qc-render")
        root.rename(held)
        root.mkdir(mode=0o700)
        foreign = root / target.name
        foreign.write_bytes(foreign_bytes)
        state.update(root=root, held=held, foreign=foreign)
        original(context, inputs, target)

    monkeypatch.setattr(qc_module, "_render_bet", swap_before_render)
    try:
        with pytest.raises(QCError, match="temporary|binding|changed"):
            generate_stage_qc(case.context, "04_bet")

        assert state["foreign"].read_bytes() == foreign_bytes
        assert not state["held"].exists()
        assert list(case.context.output_directory.iterdir()) == []
    finally:
        for key in ("held", "root"):
            candidate = state.get(key)
            if candidate is not None and candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_qc_destination_swap_after_preflight_never_writes_foreign_root(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original = qc_module._render_bet
    destination = case.context.output_directory
    held = destination.with_name(f"{destination.name}.held-destination")

    def swap_destination_after_render(context, inputs, path):
        original(context, inputs, path)
        destination.rename(held)
        destination.mkdir(mode=0o700)

    monkeypatch.setattr(
        qc_module, "_render_bet", swap_destination_after_render
    )
    try:
        with pytest.raises(QCError, match="destination.*replaced|binding"):
            generate_stage_qc(case.context, "04_bet")

        assert destination.exists() and list(destination.iterdir()) == []
        assert held.exists() and list(held.iterdir()) == []
    finally:
        for candidate in (held, destination):
            if candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_qc_destination_swap_mid_copy_rolls_back_pinned_original(
    subject_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    destination = case.context.output_directory
    held = destination.with_name(f"{destination.name}.held-mid-copy")
    real_write_all = qc_module._write_all
    swapped = False

    def write_then_swap(descriptor, content):
        nonlocal swapped
        real_write_all(descriptor, content)
        if not swapped:
            swapped = True
            destination.rename(held)
            destination.mkdir(mode=0o700)

    monkeypatch.setattr(qc_module, "_write_all", write_then_swap)
    try:
        with pytest.raises(QCError, match="destination.*replaced|binding"):
            generate_stage_qc(case.context, "04_bet")

        assert swapped
        assert destination.exists() and list(destination.iterdir()) == []
        assert held.exists() and list(held.iterdir()) == []
    finally:
        for candidate in (held, destination):
            if candidate.exists():
                os.chmod(candidate, 0o700)
                shutil.rmtree(candidate)


def test_private_temp_cleanup_preserves_foreign_child_replacement(
    tmp_path: Path,
) -> None:
    import dmri_pipeline.qc as qc_module

    owner = qc_module._private_temp(
        tmp_path,
        names=("derived.bin",),
    )
    root = owner.path
    derived = root / "derived.bin"
    renamed = root / "renamed-owned.bin"
    foreign_bytes = b"foreign replacement must survive\n"
    try:
        derived.write_bytes(b"derived bytes must be erased\n")
        os.chmod(root, 0o700)
        derived.rename(renamed)
        derived.write_bytes(foreign_bytes)

        owner.close()
        owner.close()

        assert derived.read_bytes() == foreign_bytes
        assert not renamed.exists()
    finally:
        if root.exists():
            os.chmod(root, 0o700)
            shutil.rmtree(root)


def test_commit_rollback_preserves_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first-owned\n", encoding="utf-8")
    second.write_text("second-owned\n", encoding="utf-8")
    real_write_all = qc_module._write_all
    calls = 0

    def replace_then_fail(descriptor, content):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write_all(descriptor, content)
        target = destination / "first.txt"
        target.unlink()
        target.write_text("foreign replacement\n", encoding="utf-8")
        raise OSError("injected second-output failure")

    monkeypatch.setattr(qc_module, "_write_all", replace_then_fail)
    with pytest.raises(QCError, match="commit"):
        qc_module._commit_files(
            destination,
            ((first, "first.txt"), (second, "second.txt")),
            require_empty=True,
        )
    assert (destination / "first.txt").read_text(encoding="utf-8") == (
        "foreign replacement\n"
    )
    assert not (destination / "second.txt").exists()


def test_commit_runtime_error_after_copy_rolls_back_and_preserves_type(
    tmp_path: Path,
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"derived output bytes\n")
    verifier_calls = 0

    def fail_after_copy() -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 2:
            raise RuntimeError("post-copy failure probe")

    with pytest.raises(RuntimeError, match="post-copy failure probe"):
        qc_module._commit_files(
            destination,
            ((source, "output.bin"),),
            require_empty=True,
            verifier=fail_after_copy,
        )

    assert list(destination.iterdir()) == []


def test_commit_post_open_fstat_failure_cleans_pending_output_and_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"derived output bytes\n")
    real_open = qc_module.os.open
    real_fstat = qc_module.os.fstat
    state: dict[str, object] = {}

    def track_output_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "output.bin" and dir_fd is not None:
            state["descriptor"] = descriptor
        return descriptor

    def fail_once(descriptor):
        if (
            descriptor == state.get("descriptor")
            and not state.get("failed")
        ):
            state["failed"] = True
            raise OSError("injected post-open fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(qc_module.os, "open", track_output_open)
    monkeypatch.setattr(qc_module.os, "fstat", fail_once)
    with pytest.raises(QCError, match="commit"):
        qc_module._commit_files(
            destination,
            ((source, "output.bin"),),
            require_empty=True,
        )

    assert list(destination.iterdir()) == []
    with pytest.raises(OSError):
        real_fstat(int(state["descriptor"]))


def test_commit_post_open_keyboard_interrupt_cleans_pending_output_and_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"derived output bytes\n")
    real_open = qc_module.os.open
    real_fstat = qc_module.os.fstat
    state: dict[str, object] = {}

    def track_output_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "output.bin" and dir_fd is not None:
            state["descriptor"] = descriptor
        return descriptor

    def interrupt_once(descriptor):
        if (
            descriptor == state.get("descriptor")
            and not state.get("failed")
        ):
            state["failed"] = True
            raise KeyboardInterrupt("injected post-open interrupt")
        return real_fstat(descriptor)

    monkeypatch.setattr(qc_module.os, "open", track_output_open)
    monkeypatch.setattr(qc_module.os, "fstat", interrupt_once)
    with pytest.raises(KeyboardInterrupt, match="post-open interrupt"):
        qc_module._commit_files(
            destination,
            ((source, "output.bin"),),
            require_empty=True,
        )

    assert list(destination.iterdir()) == []
    with pytest.raises(OSError):
        real_fstat(int(state["descriptor"]))


def test_commit_keyboard_interrupt_after_copy_rolls_back_and_preserves_type(
    tmp_path: Path,
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"derived output bytes\n")
    verifier_calls = 0

    def interrupt_after_copy() -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 2:
            raise KeyboardInterrupt("post-copy interrupt probe")

    with pytest.raises(KeyboardInterrupt, match="interrupt probe"):
        qc_module._commit_files(
            destination,
            ((source, "output.bin"),),
            require_empty=True,
            verifier=interrupt_after_copy,
        )

    assert list(destination.iterdir()) == []


def test_commit_rollback_retries_first_unlink_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"derived output bytes\n")
    verifier_calls = 0
    unlink_calls = 0
    real_unlink = qc_module.os.unlink

    def fail_after_copy() -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 2:
            raise RuntimeError("post-copy transient-unlink probe")

    def fail_first_unlink(path, *, dir_fd=None):
        nonlocal unlink_calls
        if dir_fd is not None:
            unlink_calls += 1
            if unlink_calls == 1:
                raise OSError("transient unlink failure")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(qc_module.os, "unlink", fail_first_unlink)
    with pytest.raises(RuntimeError, match="transient-unlink probe"):
        qc_module._commit_files(
            destination,
            ((source, "output.bin"),),
            require_empty=True,
            verifier=fail_after_copy,
        )

    assert unlink_calls >= 2
    assert list(destination.iterdir()) == []


def test_commit_persistent_cleanup_failure_is_visible_on_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"derived output bytes\n")
    verifier_calls = 0
    real_unlink = qc_module.os.unlink

    def fail_after_copy() -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 2:
            raise RuntimeError("post-copy persistent-cleanup probe")

    def fail_owned_unlink(path, *, dir_fd=None):
        if dir_fd is not None:
            raise OSError("persistent unlink failure")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(qc_module.os, "unlink", fail_owned_unlink)
    try:
        with pytest.raises(
            RuntimeError, match="persistent-cleanup probe"
        ) as captured:
            qc_module._commit_files(
                destination,
                ((source, "output.bin"),),
                require_empty=True,
                verifier=fail_after_copy,
            )

        notes = getattr(captured.value, "__notes__", ())
        assert any("rollback cleanup" in note for note in notes)
        output = destination / "output.bin"
        assert output.exists() and output.stat().st_size == 0
    finally:
        monkeypatch.setattr(qc_module.os, "unlink", real_unlink)
        output = destination / "output.bin"
        if output.exists():
            output.unlink()


def test_input_replacement_at_commit_boundary_is_rejected(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original_commit = qc_module._commit_files

    def replace_input(destination, sources, **kwargs):
        replacement = case.context.bvals.with_name("replacement.bval")
        replacement.write_bytes(case.context.bvals.read_bytes())
        os.replace(replacement, case.context.bvals)
        return original_commit(destination, sources, **kwargs)

    monkeypatch.setattr(qc_module, "_commit_files", replace_input)
    with pytest.raises(QCError, match="replaced"):
        generate_stage_qc(case.context, "04_bet")
    assert list(case.context.output_directory.iterdir()) == []


def test_qc_detects_modify_then_restore_of_declared_input(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original = qc_module._render_bet

    def mutate_restore(context, inputs, path):
        original(context, inputs, path)
        exact = context.brain_mask.read_bytes()
        context.brain_mask.write_bytes(exact + b"x")
        context.brain_mask.write_bytes(exact)

    monkeypatch.setattr(qc_module, "_render_bet", mutate_restore)
    with pytest.raises(QCError, match="changed"):
        generate_stage_qc(case.context, "04_bet")
    assert list(case.context.output_directory.iterdir()) == []


def test_qc_rejects_external_hardlink_even_without_declared_alias(
    subject_config,
) -> None:
    case = make_qc_case(subject_config)
    external = case.context.brain_mask.with_name("external_mask_alias.nii.gz")
    os.link(case.context.brain_mask, external)
    with pytest.raises(QCError, match="link count"):
        generate_stage_qc(case.context, "04_bet")


def test_full_grid_binary_mask_has_visible_zero_border_boundary() -> None:
    import dmri_pipeline.qc as qc_module

    boundary = qc_module._binary_boundary(np.ones((8, 8), dtype=bool))
    assert boundary.shape == (8, 8)
    assert np.all(boundary[[0, -1], :])
    assert np.all(boundary[:, [0, -1]])
    assert not boundary[4, 4]


def test_qc_rejects_transient_snapshot_symlink_swap(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    case = make_qc_case(subject_config)
    original = qc_module._render_bet

    def transient_swap(context, inputs, path):
        snapshot = inputs.snapshots[context.hifi_b0]
        held = snapshot.with_name(f"{snapshot.name}.held")
        os.chmod(snapshot.parent, 0o700)
        snapshot.rename(held)
        try:
            snapshot.symlink_to(context.hifi_b0)
            original(context, inputs, path)
        finally:
            snapshot.unlink()
            held.rename(snapshot)
            os.chmod(snapshot.parent, 0o500)

    monkeypatch.setattr(qc_module, "_render_bet", transient_swap)
    with pytest.raises(QCError, match="snapshot directory changed"):
        generate_stage_qc(case.context, "04_bet")
    assert list(case.context.output_directory.iterdir()) == []


def test_input_guard_cleanup_uses_owned_inodes_after_root_path_reuse(
    subject_config,
) -> None:
    import dmri_pipeline.qc as qc_module

    scientific_bytes = subject_config.bvals.read_bytes()
    guard = qc_module._InputGuard([subject_config.bvals])
    guard.__enter__()
    snapshot_root = guard._snapshot_root
    assert snapshot_root is not None
    owned_snapshots = tuple(guard.paths.values())
    held_root = snapshot_root.with_name(f"{snapshot_root.name}.held-test")
    foreign_bytes = b"foreign tree must survive byte-identical\n"
    try:
        os.chmod(snapshot_root, 0o700)
        snapshot_root.rename(held_root)
        (snapshot_root / "nested").mkdir(parents=True, mode=0o700)
        foreign_file = snapshot_root / "nested" / "foreign.bin"
        foreign_file.write_bytes(foreign_bytes)

        guard.close()

        assert foreign_file.read_bytes() == foreign_bytes
        assert not held_root.exists() or list(held_root.iterdir()) == []
        assert all(not path.exists() for path in owned_snapshots)
        if held_root.exists():
            assert all(
                path.read_bytes() != scientific_bytes
                for path in held_root.rglob("*")
                if path.is_file()
            )
    finally:
        guard.close()
        for path in (held_root, snapshot_root):
            if path.exists():
                os.chmod(path, 0o700)
                shutil.rmtree(path)


def test_input_guard_ordinary_cleanup_removes_all_owned_snapshots(
    subject_config,
) -> None:
    import dmri_pipeline.qc as qc_module

    guard = qc_module._InputGuard([subject_config.bvals])
    guard.__enter__()
    snapshot_root = guard._snapshot_root
    snapshots = tuple(guard.paths.values())
    assert snapshot_root is not None and snapshot_root.exists()

    guard.close()

    assert not snapshot_root.exists()
    assert all(not path.exists() for path in snapshots)


def test_input_guard_root_open_failure_preserves_unpinned_empty_root(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    created: dict[str, object] = {}
    real_mkdir = qc_module.os.mkdir
    real_open = qc_module.os.open

    def tracked_mkdir(path, mode=0o777, *, dir_fd=None):
        result = real_mkdir(path, mode, dir_fd=dir_fd)
        if dir_fd is not None and str(path).startswith("dmri-qc-inputs-"):
            created["name"] = str(path)
            created["root"] = (
                Path(qc_module.tempfile.gettempdir()).resolve() / str(path)
            )
        return result

    def fail_root_open(path, flags, mode=0o777, *, dir_fd=None):
        if (
            dir_fd is not None
            and path == created.get("name")
        ):
            raise OSError("injected snapshot-root open failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(qc_module.os, "mkdir", tracked_mkdir)
    monkeypatch.setattr(qc_module.os, "open", fail_root_open)

    with pytest.raises(OSError, match="snapshot-root open failure"):
        qc_module._InputGuard([subject_config.bvals]).__enter__()

    root = created["root"]
    assert isinstance(root, Path)
    assert root.exists() and list(root.iterdir()) == []
    root.rmdir()


def test_open_regular_read_closes_descriptor_when_fstat_fails(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    real_open = qc_module.os.open
    real_fstat = qc_module.os.fstat
    opened: list[int] = []
    failed = False

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def fail_first_fstat(descriptor):
        nonlocal failed
        if descriptor in opened and not failed:
            failed = True
            raise OSError("injected source fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(qc_module.os, "open", tracked_open)
    monkeypatch.setattr(qc_module.os, "fstat", fail_first_fstat)
    with pytest.raises(QCError, match="inspect required input"):
        qc_module._open_regular_read(subject_config.bvals)

    assert opened
    with pytest.raises(OSError):
        real_fstat(opened[-1])


def test_input_guard_post_copy_handoff_failure_removes_snapshot_bytes(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    scientific_bytes = subject_config.bvals.read_bytes()
    real_copy = qc_module._copy_descriptor_snapshot
    real_fstat = qc_module.os.fstat
    state: dict[str, object] = {}
    guard = qc_module._InputGuard([subject_config.bvals])

    def tracked_copy(*args, **kwargs):
        result = real_copy(*args, **kwargs)
        state["snapshot_descriptor"] = result[1]
        state["snapshot_root"] = guard._snapshot_root
        return result

    def fail_handoff_fstat(descriptor):
        if (
            descriptor == state.get("snapshot_descriptor")
            and not state.get("failed")
        ):
            state["failed"] = True
            raise OSError("injected post-copy handoff failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(qc_module, "_copy_descriptor_snapshot", tracked_copy)
    monkeypatch.setattr(qc_module.os, "fstat", fail_handoff_fstat)

    with pytest.raises(OSError, match="post-copy handoff failure"):
        guard.__enter__()

    snapshot_root = state["snapshot_root"]
    assert isinstance(snapshot_root, Path)
    assert not snapshot_root.exists() or list(snapshot_root.iterdir()) == []
    if snapshot_root.exists():
        assert all(
            path.read_bytes() != scientific_bytes
            for path in snapshot_root.rglob("*")
            if path.is_file()
        )
    with pytest.raises(OSError):
        real_fstat(int(state["snapshot_descriptor"]))


def test_input_guard_cleanup_failure_propagates_and_can_be_retried(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    guard = qc_module._InputGuard([subject_config.bvals])
    guard.__enter__()
    snapshot_root = guard._snapshot_root
    root_descriptor = guard._snapshot_directory_descriptor
    snapshots = tuple(guard.paths.values())
    assert snapshot_root is not None and root_descriptor is not None
    real_unlink = qc_module.os.unlink

    def fail_owned_unlink(path, *, dir_fd=None):
        if dir_fd == root_descriptor:
            raise OSError("injected owned-unlink failure")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(qc_module.os, "unlink", fail_owned_unlink)
    try:
        with pytest.raises(QCError, match="snapshot cleanup"):
            guard.close()
        assert all(path.exists() and path.stat().st_size == 0 for path in snapshots)

        monkeypatch.setattr(qc_module.os, "unlink", real_unlink)
        guard.close()
        assert not snapshot_root.exists()
    finally:
        monkeypatch.setattr(qc_module.os, "unlink", real_unlink)
        try:
            guard.close()
        except QCError:
            pass
        if snapshot_root.exists():
            os.chmod(snapshot_root, 0o700)
            shutil.rmtree(snapshot_root)


def test_input_guard_cleanup_failure_does_not_mask_primary_error(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dmri_pipeline.qc as qc_module

    guard = qc_module._InputGuard([subject_config.bvals])
    real_unlink = qc_module.os.unlink
    state: dict[str, object] = {}

    def fail_owned_unlink(path, *, dir_fd=None):
        if dir_fd == state.get("root_descriptor"):
            raise OSError("injected owned-unlink failure")
        return real_unlink(path, dir_fd=dir_fd)

    try:
        with pytest.raises(RuntimeError, match="primary operation failure"):
            with guard as entered:
                state["root_descriptor"] = (
                    entered._snapshot_directory_descriptor
                )
                state["snapshot_root"] = entered._snapshot_root
                monkeypatch.setattr(
                    qc_module.os, "unlink", fail_owned_unlink
                )
                raise RuntimeError("primary operation failure")
    finally:
        monkeypatch.setattr(qc_module.os, "unlink", real_unlink)
        guard.close()
        snapshot_root = state.get("snapshot_root")
        assert isinstance(snapshot_root, Path)
        if snapshot_root.exists():
            os.chmod(snapshot_root, 0o700)
            shutil.rmtree(snapshot_root)


def test_all_qc_is_deterministic_for_identical_inputs(subject_config) -> None:
    case = make_qc_case(subject_config)
    first = generate_all_qc(case.context)
    second_context = replace(
        case.context,
        output_directory=case.context.stage_context.subject_root / "qc_again",
    )
    second_context.output_directory.mkdir()
    second = generate_all_qc(second_context)

    def digest(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert [digest(first[key]) for key in first] == [
        digest(second[key]) for key in second
    ]
    assert (
        (case.context.output_directory / "qc_manifest.json").read_bytes()
        == (second_context.output_directory / "qc_manifest.json").read_bytes()
    )
