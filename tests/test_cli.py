from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import dmri_pipeline.cli as cli
import dmri_pipeline.orchestrator as orchestrator
from dmri_pipeline.cluster import ClusterConfigError
from dmri_pipeline.fsl import FSLDiscoveryError
from dmri_pipeline.noddi import NODDIError
from dmri_pipeline.orchestrator import PipelineOutcome
from dmri_pipeline.preprocess import PreprocessError


def test_cli_prints_nonmutating_cluster_context(
    subject_config, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Cluster context mode emits only wrapper inputs and never runs the pipeline."""
    configured = replace(
        subject_config,
        analysis=replace(subject_config.analysis, noddi_workers=3),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: configured)

    def pipeline_must_not_run(*args, **kwargs):
        raise AssertionError("context mode must not run the pipeline")

    monkeypatch.setattr(cli, "run_pipeline", pipeline_must_not_run)

    assert cli.main(["--print-cluster-context", str(configured.config_path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "subject_id": "SYNTH001",
        "subject_output": str(configured.output_root / "SYNTH001"),
        "noddi_workers": 3,
    }
    assert list(payload) == ["noddi_workers", "subject_id", "subject_output"]
    assert not configured.subject_output.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--stage-group", "topup", "--print-cluster-context", "subject.yaml"],
        [
            "--force-stage",
            "00_input_audit",
            "--print-cluster-context",
            "subject.yaml",
        ],
        ["--dry-run", "--print-cluster-context", "subject.yaml"],
        ["--validate-only", "--print-cluster-context", "subject.yaml"],
    ],
)
def test_cli_rejects_cluster_context_with_execution_modes(arguments, capsys) -> None:
    """Context printing is standalone so wrappers receive unambiguous values."""
    assert cli.main(arguments) == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_cli_maps_cluster_context_validation_error_to_exit_2(
    subject_config, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Invalid cluster wrapper context is a user configuration error."""
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)

    def fail_context(config):
        raise ClusterConfigError("explicit worker count required")

    monkeypatch.setattr(cli, "cluster_subject_context", fail_context)

    assert cli.main(["--print-cluster-context", str(subject_config.config_path)]) == 2
    assert "explicit worker count required" in capsys.readouterr().err


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
        ("GROUP_COMPLETE", 0),
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
        lambda config, mode, force_stage=None, stage_group=None: PipelineOutcome(
            config.subject_id, status, (), config.subject_output
        ),
    )

    assert cli.main([str(subject_config.config_path)]) == exit_code


def test_cli_passes_stage_group_by_keyword_and_prints_group(
    subject_config, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    captured_call: tuple[object, str, str | None, str | None] | None = None
    monkeypatch.setattr(cli, "load_config", lambda path: subject_config)

    def fake_run_pipeline(
        config, *, mode: str, force_stage: str | None = None, stage_group: str | None = None
    ) -> PipelineOutcome:
        nonlocal captured_call
        captured_call = (config, mode, force_stage, stage_group)
        return PipelineOutcome(
            config.subject_id, "GROUP_COMPLETE", (), config.subject_output
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main(["--stage-group", "topup", str(subject_config.config_path)]) == 0
    assert captured_call == (subject_config, "run", None, "topup")
    assert "group=topup" in capsys.readouterr().out


def test_cli_rejects_unknown_stage_group(capsys) -> None:
    assert cli.main(["--stage-group", "unknown", "subject.yaml"]) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_rejects_validate_only_stage_group(capsys) -> None:
    assert cli.main(["--validate-only", "--stage-group", "topup", "subject.yaml"]) == 2
    assert "--stage-group cannot be combined with --validate-only" in capsys.readouterr().err


def test_cli_rejects_force_stage_outside_selected_group(capsys) -> None:
    assert (
        cli.main(
            [
                "--stage-group",
                "topup",
                "--force-stage",
                "05_eddy",
                "subject.yaml",
            ]
        )
        == 2
    )
    assert "outside stage group" in capsys.readouterr().err


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
