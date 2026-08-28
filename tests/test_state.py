from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
from pathlib import Path

import pytest

import dmri_pipeline.state as state_module
from dmri_pipeline.state import (
    StageContext,
    StageRecord,
    StageRunner,
    StageSpec,
    StageStateError,
    stage_signature,
)


@pytest.fixture
def stage_context(subject_config) -> StageContext:
    return StageContext(
        config=subject_config,
        package_root=Path(__file__).parents[1],
        subject_root=subject_config.subject_output,
        software={"python": "3.11.13", "matlab": "R2025a", "fsl": "6.0.7.18"},
    )


@pytest.fixture
def stage_runner(stage_context: StageContext) -> StageRunner:
    return StageRunner(stage_context)


def stage_spec_that_writes(
    filename: str = "result.txt",
    *,
    input_paths: tuple[Path, ...] = (),
    source_paths: tuple[Path, ...] = (),
    resource_paths: tuple[Path, ...] = (),
    calls: list[Path] | None = None,
) -> StageSpec:
    def action(work_dir: Path) -> None:
        if calls is not None:
            calls.append(work_dir)
        (work_dir / filename).write_text("ok\n", encoding="utf-8")

    def validator(work_dir: Path) -> tuple[Path, ...]:
        return (work_dir / filename,)

    return StageSpec(
        name="01_test",
        action=action,
        validator=validator,
        input_paths=input_paths,
        source_paths=source_paths,
        resource_paths=resource_paths,
    )


def test_success_promotes_work_directory_atomically(stage_runner: StageRunner) -> None:
    calls: list[Path] = []
    outcome = stage_runner.run(stage_spec_that_writes(calls=calls))

    assert outcome.status == "completed"
    assert (outcome.directory / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert calls == [stage_runner.work_dir("01_test")]
    assert not stage_runner.work_dir("01_test").exists()
    assert outcome.record_path == stage_runner.record_path("01_test")


def test_failure_preserves_work_and_writes_no_completion_record(
    stage_runner: StageRunner,
) -> None:
    def fail(work_dir: Path) -> None:
        (work_dir / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("stage failed")

    spec = StageSpec("01_test", fail, lambda work: (), (), ())

    with pytest.raises(RuntimeError, match="stage failed"):
        stage_runner.run(spec)

    assert (stage_runner.work_dir("01_test") / "partial.txt").exists()
    assert not stage_runner.record_path("01_test").exists()


def test_validator_failure_preserves_work(
    stage_runner: StageRunner,
) -> None:
    def action(work_dir: Path) -> None:
        (work_dir / "partial.txt").write_text("partial\n", encoding="utf-8")

    def reject(work_dir: Path) -> tuple[Path, ...]:
        raise ValueError("invalid output")

    with pytest.raises(ValueError, match="invalid output"):
        stage_runner.run(StageSpec("01_test", action, reject, (), ()))

    assert (stage_runner.work_dir("01_test") / "partial.txt").exists()
    assert not stage_runner.record_path("01_test").exists()


def test_missing_validated_output_preserves_work(stage_runner: StageRunner) -> None:
    spec = StageSpec(
        "01_test",
        lambda work: None,
        lambda work: (work / "missing.txt",),
        (),
        (),
    )

    with pytest.raises(StageStateError, match="required output"):
        stage_runner.run(spec)

    assert stage_runner.work_dir("01_test").exists()
    assert not stage_runner.record_path("01_test").exists()


def test_completion_record_is_deterministic_and_parseable(
    stage_runner: StageRunner, subject_config
) -> None:
    spec = stage_spec_that_writes(
        input_paths=(subject_config.bvecs, subject_config.bvals)
    )
    outcome = stage_runner.run(spec)
    text = outcome.record_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    record = StageRecord.from_dict(payload)

    assert record.to_dict() == payload
    assert payload["stage"] == "01_test"
    assert payload["subject_id"] == "SYNTH001"
    assert payload["package_version"] == "2.1.0"
    assert payload["config_sha256"] == hashlib.sha256(
        json.dumps(
            subject_config.canonical_dict(),
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert payload["outputs"] == [
        {
            "relative_path": "result.txt",
            "sha256": hashlib.sha256(b"ok\n").hexdigest(),
            "size": 3,
        }
    ]
    assert [item["path"] for item in payload["inputs"]] == [
        "inputs/pa_dwi.bval",
        "inputs/pa_dwi.bvec",
    ]
    assert payload["software"] == {
        "fsl": "6.0.7.18",
        "matlab": "R2025a",
        "python": "3.11.13",
    }
    assert text == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_matching_record_returns_skipped_without_rerunning_action(
    stage_runner: StageRunner,
) -> None:
    calls: list[Path] = []
    spec = stage_spec_that_writes(calls=calls)
    completed = stage_runner.run(spec)
    skipped = stage_runner.run(spec)

    assert completed.status == "completed"
    assert skipped.status == "skipped"
    assert skipped.directory == completed.directory
    assert calls == [stage_runner.work_dir("01_test")]


def test_changed_config_invalidates_resume(
    stage_runner: StageRunner, stage_context: StageContext
) -> None:
    spec = stage_spec_that_writes()
    stage_runner.run(spec)
    changed_analysis = replace(
        stage_context.config.analysis,
        dti_max_b=stage_context.config.analysis.dti_max_b + 100,
    )
    changed_config = replace(stage_context.config, analysis=changed_analysis)
    changed_context = replace(stage_context, config=changed_config)

    assert not StageRunner(changed_context).is_current(spec)


def test_changed_software_provenance_invalidates_resume(
    stage_runner: StageRunner, stage_context: StageContext
) -> None:
    spec = stage_spec_that_writes()
    stage_runner.run(spec)
    changed_context = replace(
        stage_context,
        software={**stage_context.software, "fsl": "6.0.7.19"},
    )

    assert not StageRunner(changed_context).is_current(spec)


def test_changed_source_contents_invalidate(
    stage_runner: StageRunner, tmp_path: Path
) -> None:
    source = tmp_path / "stage.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    spec = stage_spec_that_writes(source_paths=(source,))
    stage_runner.run(spec)

    source.write_text("VERSION = 2\n", encoding="utf-8")

    assert not stage_runner.is_current(spec)


def test_changed_resource_contents_invalidate(
    stage_runner: StageRunner, tmp_path: Path
) -> None:
    resource = tmp_path / "atlas.txt"
    resource.write_text("atlas-v1\n", encoding="utf-8")
    spec = stage_spec_that_writes(resource_paths=(resource,))
    stage_runner.run(spec)

    resource.write_text("atlas-v2\n", encoding="utf-8")

    assert not stage_runner.is_current(spec)


def test_changed_input_contents_invalidate(
    stage_runner: StageRunner, subject_config
) -> None:
    spec = stage_spec_that_writes(input_paths=(subject_config.bvals,))
    stage_runner.run(spec)

    subject_config.bvals.write_text("changed\n", encoding="utf-8")

    assert not stage_runner.is_current(spec)


def test_missing_output_invalidates(stage_runner: StageRunner) -> None:
    spec = stage_spec_that_writes()
    outcome = stage_runner.run(spec)

    (outcome.directory / "result.txt").unlink()

    assert not stage_runner.is_current(spec)


def test_changed_output_size_invalidates(stage_runner: StageRunner) -> None:
    spec = stage_spec_that_writes()
    outcome = stage_runner.run(spec)

    (outcome.directory / "result.txt").write_text("changed\n", encoding="utf-8")

    assert not stage_runner.is_current(spec)


def test_changed_output_contents_with_same_size_invalidate(
    stage_runner: StageRunner,
) -> None:
    spec = stage_spec_that_writes()
    outcome = stage_runner.run(spec)

    (outcome.directory / "result.txt").write_text("NO\n", encoding="utf-8")

    assert not stage_runner.is_current(spec)


def test_stale_final_directory_raises_rather_than_overwrites(
    stage_runner: StageRunner,
) -> None:
    final_dir = stage_runner.final_dir("01_test")
    final_dir.mkdir(parents=True)
    (final_dir / "old.txt").write_text("keep\n", encoding="utf-8")
    calls: list[Path] = []

    with pytest.raises(StageStateError, match=r"force|invalidate"):
        stage_runner.run(stage_spec_that_writes(calls=calls))

    assert (final_dir / "old.txt").read_text(encoding="utf-8") == "keep\n"
    assert calls == []


def test_final_appearing_immediately_before_promotion_is_not_clobbered(
    stage_runner: StageRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_dir = (
        stage_runner.context.subject_root / "01_test"
    )
    real_entry_exists = state_module._entry_exists
    final_checks = 0

    def inject_race(path: Path) -> bool:
        nonlocal final_checks
        if path == final_dir:
            final_checks += 1
            if final_checks == 2:
                final_dir.mkdir(parents=True)
                (final_dir / "racer.txt").write_text("preserve\n", encoding="utf-8")
                return False
        return real_entry_exists(path)

    monkeypatch.setattr(state_module, "_entry_exists", inject_race)

    with pytest.raises(StageStateError, match=r"appeared|promot"):
        stage_runner.run(stage_spec_that_writes())

    assert (final_dir / "racer.txt").read_text(encoding="utf-8") == "preserve\n"
    assert not (final_dir / "result.txt").exists()
    assert (
        stage_runner.context.subject_root / ".work" / "01_test" / "result.txt"
    ).exists()


def _make_failing_linux_libc(error_number: int):
    class FailingRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *args) -> int:
            state_module.ctypes.set_errno(error_number)
            return -1

    class FakeLibc:
        renameat2 = FailingRenameAt2()

    return FakeLibc()


@pytest.mark.parametrize("unsupported_errno", [errno.ENOTSUP, errno.EINVAL])
def test_linux_nfs_unsupported_renameat2_promotes_without_clobbering(
    stage_runner: StageRunner,
    monkeypatch: pytest.MonkeyPatch,
    unsupported_errno: int,
) -> None:
    monkeypatch.setattr(state_module.sys, "platform", "linux")
    monkeypatch.setattr(
        state_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _make_failing_linux_libc(unsupported_errno),
    )

    outcome = stage_runner.run(stage_spec_that_writes())

    assert outcome.status == "completed"
    assert (outcome.directory / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert outcome.record_path.is_file()
    assert not stage_runner.work_dir("01_test").exists()


def test_linux_missing_renameat2_uses_directory_reservation(
    stage_runner: StageRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LibcWithoutRenameAt2:
        pass

    monkeypatch.setattr(state_module.sys, "platform", "linux")
    monkeypatch.setattr(
        state_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: LibcWithoutRenameAt2(),
    )

    outcome = stage_runner.run(stage_spec_that_writes())

    assert outcome.status == "completed"
    assert (outcome.directory / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert outcome.record_path.is_file()
    assert not stage_runner.work_dir("01_test").exists()


@pytest.mark.parametrize("unsupported_errno", [errno.ENOTSUP, errno.EINVAL])
def test_linux_nfs_fallback_preserves_empty_final_directory_race(
    stage_runner: StageRunner,
    monkeypatch: pytest.MonkeyPatch,
    unsupported_errno: int,
) -> None:
    final_dir = stage_runner.final_dir("01_test")
    real_entry_exists = state_module._entry_exists
    final_checks = 0
    raced_inode: int | None = None

    def inject_empty_race(path: Path) -> bool:
        nonlocal final_checks, raced_inode
        if path == final_dir:
            final_checks += 1
            if final_checks == 2:
                final_dir.mkdir(parents=True)
                raced_inode = final_dir.stat().st_ino
                return False
        return real_entry_exists(path)

    monkeypatch.setattr(state_module.sys, "platform", "linux")
    monkeypatch.setattr(
        state_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _make_failing_linux_libc(unsupported_errno),
    )
    monkeypatch.setattr(state_module, "_entry_exists", inject_empty_race)

    with pytest.raises(StageStateError, match=r"appeared during promotion"):
        stage_runner.run(stage_spec_that_writes())

    assert raced_inode is not None
    assert final_dir.stat().st_ino == raced_inode
    assert list(final_dir.iterdir()) == []
    work_dir = stage_runner.work_dir("01_test")
    assert (work_dir / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert (work_dir / ".stage_complete.json").is_file()


def test_linux_nfs_fallback_reports_populated_reservation_as_race(
    stage_runner: StageRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_dir = stage_runner.final_dir("01_test")
    real_replace = state_module.os.replace

    def populate_and_reject(source: Path, destination: Path) -> None:
        if Path(destination) != final_dir:
            real_replace(source, destination)
            return
        (destination / "racer.txt").write_text("preserve\n", encoding="utf-8")
        raise OSError(errno.ENOTEMPTY, "destination is no longer empty", destination)

    monkeypatch.setattr(state_module.sys, "platform", "linux")
    monkeypatch.setattr(
        state_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _make_failing_linux_libc(errno.ENOTSUP),
    )
    monkeypatch.setattr(state_module.os, "replace", populate_and_reject)

    with pytest.raises(StageStateError, match=r"appeared during promotion"):
        stage_runner.run(stage_spec_that_writes())

    assert (final_dir / "racer.txt").read_text(encoding="utf-8") == "preserve\n"
    work_dir = stage_runner.work_dir("01_test")
    assert (work_dir / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert (work_dir / ".stage_complete.json").is_file()


@pytest.mark.parametrize("reported_errno", [errno.EIO, errno.EEXIST, errno.ENOTEMPTY])
def test_linux_nfs_reconciles_rename_that_succeeded_but_reported_error(
    stage_runner: StageRunner,
    monkeypatch: pytest.MonkeyPatch,
    reported_errno: int,
) -> None:
    final_dir = stage_runner.final_dir("01_test")
    real_replace = state_module.os.replace

    def replace_then_report_error(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        if Path(destination) != final_dir:
            return
        raise OSError(reported_errno, "simulated lost NFS reply", destination)

    monkeypatch.setattr(state_module.sys, "platform", "linux")
    monkeypatch.setattr(
        state_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _make_failing_linux_libc(errno.ENOTSUP),
    )
    monkeypatch.setattr(state_module.os, "replace", replace_then_report_error)

    outcome = stage_runner.run(stage_spec_that_writes())

    assert outcome.status == "completed"
    assert (outcome.directory / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert outcome.record_path.is_file()
    assert not stage_runner.work_dir("01_test").exists()


def test_linux_nfs_fallback_does_not_mask_unrelated_rename_error(
    stage_runner: StageRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state_module.sys, "platform", "linux")
    monkeypatch.setattr(
        state_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _make_failing_linux_libc(errno.EIO),
    )

    with pytest.raises(StageStateError, match=r"Cannot atomically promote"):
        stage_runner.run(stage_spec_that_writes())

    assert not stage_runner.final_dir("01_test").exists()
    work_dir = stage_runner.work_dir("01_test")
    assert (work_dir / "result.txt").read_text(encoding="utf-8") == "ok\n"
    assert (work_dir / ".stage_complete.json").is_file()


def test_invalidate_from_archives_affected_final_and_work_directories(
    stage_runner: StageRunner,
) -> None:
    first = stage_runner.final_dir("01_first")
    second = stage_runner.final_dir("02_second")
    second_work = stage_runner.work_dir("02_second")
    third_work = stage_runner.work_dir("03_third")
    for directory in (first, second, second_work, third_work):
        directory.mkdir(parents=True)
        (directory / "marker.txt").write_text(directory.name, encoding="utf-8")

    stage_runner.invalidate_from(
        ("01_first", "02_second", "03_third"), "02_second"
    )

    assert first.exists()
    assert not second.exists()
    assert not second_work.exists()
    assert not third_work.exists()
    archives = list((stage_runner.context.subject_root / ".invalidated").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "final" / "02_second" / "marker.txt").exists()
    assert (archives[0] / "work" / "02_second" / "marker.txt").exists()
    assert (archives[0] / "work" / "03_third" / "marker.txt").exists()


@pytest.mark.parametrize(
    "name", ["", "a/b", "../escape", "stage name", ".hidden", r"a\b"]
)
def test_unsafe_stage_names_reject(name: str) -> None:
    with pytest.raises(StageStateError, match="stage name"):
        StageSpec(name, lambda work: None, lambda work: (), (), ())


def test_symlinked_work_root_cannot_point_action_outside_subject_root(
    stage_runner: StageRunner, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    subject_root = stage_runner.context.subject_root
    subject_root.mkdir(parents=True)
    (subject_root / ".work").symlink_to(outside, target_is_directory=True)
    called = False

    def action(work_dir: Path) -> None:
        nonlocal called
        called = True

    spec = StageSpec("01_test", action, lambda work: (), (), ())
    with pytest.raises(StageStateError, match="outside subject root"):
        stage_runner.run(spec)

    assert not called


def test_in_subject_symlinked_work_stage_cannot_alias_another_stage(
    stage_runner: StageRunner,
) -> None:
    subject_root = stage_runner.context.subject_root
    other_stage = subject_root / "02_other_stage"
    other_stage.mkdir(parents=True)
    (other_stage / "marker.txt").write_text("preserve\n", encoding="utf-8")
    work_root = subject_root / ".work"
    work_root.mkdir()
    (work_root / "01_test").symlink_to(other_stage, target_is_directory=True)
    called = False

    def action(work_dir: Path) -> None:
        nonlocal called
        called = True
        (work_dir / "result.txt").write_text("bad\n", encoding="utf-8")

    spec = StageSpec("01_test", action, lambda work: (work / "result.txt",), (), ())
    with pytest.raises(StageStateError, match="symbolic link"):
        stage_runner.run(spec)

    assert not called
    assert (other_stage / "marker.txt").read_text(encoding="utf-8") == "preserve\n"
    assert not (other_stage / "result.txt").exists()


def test_in_subject_symlinked_final_stage_is_rejected(
    stage_runner: StageRunner,
) -> None:
    subject_root = stage_runner.context.subject_root
    other_stage = subject_root / "02_other_stage"
    other_stage.mkdir(parents=True)
    (other_stage / "marker.txt").write_text("preserve\n", encoding="utf-8")
    (subject_root / "01_test").symlink_to(other_stage, target_is_directory=True)

    with pytest.raises(StageStateError, match="symbolic link"):
        stage_runner.run(stage_spec_that_writes())

    assert (other_stage / "marker.txt").read_text(encoding="utf-8") == "preserve\n"
    assert not (other_stage / "result.txt").exists()


def test_stage_context_copies_software_mapping(subject_config) -> None:
    software = {"python": "3.11"}
    context = StageContext(
        subject_config,
        Path(__file__).parents[1],
        subject_config.subject_output,
        software,
    )
    software["python"] = "changed"

    assert context.software["python"] == "3.11"
    with pytest.raises(TypeError):
        context.software["python"] = "changed"  # type: ignore[index]


def test_stage_signature_is_stable_for_identical_inputs(
    subject_config, tmp_path: Path
) -> None:
    source = tmp_path / "source.py"
    resource = tmp_path / "resource.txt"
    source.write_text("pass\n", encoding="utf-8")
    resource.write_text("fixed\n", encoding="utf-8")

    first = stage_signature(subject_config, "01_test", (source,), (resource,))
    second = stage_signature(subject_config, "01_test", (source,), (resource,))

    assert first == second
    assert len(first) == 64


def test_directory_dependency_is_rejected_explicitly(
    subject_config, tmp_path: Path
) -> None:
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    (source_directory / "stage.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(StageStateError, match="directory dependencies are unsupported"):
        stage_signature(subject_config, "01_test", (source_directory,), ())


def test_symlink_dependency_is_rejected_explicitly(
    subject_config, tmp_path: Path
) -> None:
    source = tmp_path / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    source_alias = tmp_path / "source-alias.py"
    source_alias.symlink_to(source)

    with pytest.raises(
        StageStateError, match="symbolic-link dependencies are unsupported"
    ):
        stage_signature(subject_config, "01_test", (source_alias,), ())
