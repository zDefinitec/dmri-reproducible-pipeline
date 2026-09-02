from __future__ import annotations

from pathlib import Path

import pytest

import dmri_pipeline.cli as cli
import dmri_pipeline.orchestrator as orchestrator
from dmri_pipeline.fsl import FSLDiscoveryError
from dmri_pipeline.noddi import NODDIError
from dmri_pipeline.orchestrator import PipelineOutcome
from dmri_pipeline.preprocess import PreprocessError


def test_cli_translates_missing_config_to_exit_2(capsys) -> None:
    assert cli.main(["missing.yaml"]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_cli_rejects_unknown_force_stage_without_loading_config(capsys) -> None:
    assert cli.main(["--force-stage", "not-a-stage", "subject.yaml"]) == 2
    assert "not-a-stage" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("COMPLETE", 0),
        ("VALIDATED", 0),
        ("DRY_RUN", 0),
        ("EXCLUDED", 20),
        ("HOLD_FOR_REVIEW", 21),
    ],
)
def test_cli_translates_pipeline_outcomes(
    subject_config, monkeypatch: pytest.MonkeyPatch, status: str, exit_code: int
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda config, mode, force_stage=None, *, selection=None: PipelineOutcome(
            config.subject_id, status, (), config.subject_output
        ),
    )

    assert cli.main([str(subject_config.config_path)]) == exit_code


def test_cli_forwards_only_stage_selection(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(config, mode, force_stage=None, *, selection=None):
        captured.update(
            config=config,
            mode=mode,
            force_stage=force_stage,
            selection=selection,
        )
        return PipelineOutcome(
            config.subject_id, "STAGE_COMPLETE", (), config.subject_output
        )

    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main(["--only-stage", "05_eddy", str(subject_config.config_path)]) == 0
    assert captured["selection"] == orchestrator.StageSelection(only_stage="05_eddy")


def test_cli_forwards_stop_after_selection(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(config, mode, force_stage=None, *, selection=None):
        captured.update(
            config=config,
            mode=mode,
            force_stage=force_stage,
            selection=selection,
        )
        return PipelineOutcome(
            config.subject_id, "PARTIAL_COMPLETE", (), config.subject_output
        )

    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main(["--stop-after", "04_bet", str(subject_config.config_path)]) == 0
    assert captured["selection"] == orchestrator.StageSelection(stop_after="04_bet")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--stop-after", "04_bet", "--only-stage", "05_eddy", "subject.yaml"],
        ["--validate-only", "--stop-after", "04_bet", "subject.yaml"],
        ["--dry-run", "--only-stage", "05_eddy", "subject.yaml"],
        ["--only-stage", "05_eddy", "--force-stage", "04_bet", "subject.yaml"],
        ["--stop-after", "04_bet", "--force-stage", "05_eddy", "subject.yaml"],
    ],
)
def test_cli_rejects_illegal_bounded_option_combinations_before_pipeline(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    called = False

    def fail_if_loaded(_path: Path):
        raise AssertionError("config should not be loaded")

    def fake_run_pipeline(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr(cli, "load_config", fail_if_loaded)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main(arguments) == 2
    assert "ERROR:" in capsys.readouterr().err
    assert not called


def test_cli_accepts_forced_stage_inside_only_stage_selection(
    subject_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda config, mode, force_stage=None, *, selection=None: PipelineOutcome(
            config.subject_id, "STAGE_COMPLETE", (), config.subject_output
        ),
    )

    assert (
        cli.main(
            [
                "--only-stage",
                "05_eddy",
                "--force-stage",
                "05_eddy",
                str(subject_config.config_path),
            ]
        )
        == 0
    )


def test_cli_translates_dependency_failure_to_exit_30(
    subject_config, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FSLDiscoveryError("missing FSL")
        ),
    )

    assert cli.main([str(subject_config.config_path)]) == 30
    assert "missing FSL" in capsys.readouterr().err


def test_cli_maps_missing_package_dependency_found_before_mutation_to_30(
    subject_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)
    monkeypatch.setattr(
        orchestrator,
        "_static_package_dependencies",
        lambda: (tmp_path / "missing-package-resource",),
    )

    assert cli.main([str(subject_config.config_path)]) == 30
    assert not subject_config.subject_output.exists()


@pytest.mark.parametrize(
    ("error_factory", "exit_code"),
    [
        (lambda: orchestrator.PipelineInputError("unsafe input/output overlap"), 2),
        (lambda: orchestrator.PipelineDependencyError("missing package source"), 30),
        (lambda: PreprocessError("scientific output has wrong shape"), 50),
        (lambda: NODDIError("MATLAB command text in invalid output metadata"), 50),
        (lambda: orchestrator.PipelineOutputError("invalid stage output"), 50),
        (lambda: orchestrator.PipelineExternalError("tool exited nonzero"), 40),
    ],
)
def test_cli_uses_typed_deterministic_execution_error_mapping(
    subject_config,
    monkeypatch: pytest.MonkeyPatch,
    error_factory,
    exit_code: int,
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)

    def fail(*args, **kwargs):
        raise error_factory()

    monkeypatch.setattr(cli, "run_pipeline", fail)

    assert cli.main([str(subject_config.config_path)]) == exit_code
