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
        lambda config, mode, force_stage=None: PipelineOutcome(
            config.subject_id, status, (), config.subject_output
        ),
    )

    assert cli.main([str(subject_config.config_path)]) == exit_code


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
