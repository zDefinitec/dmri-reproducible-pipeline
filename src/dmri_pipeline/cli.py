"""Command-line entry point for the one-subject dMRI pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .audit import InputAuditError
from .config import ConfigError, load_config
from .fsl import ExternalCommandError, FSLDiscoveryError
from .models import ModelInputError, ModelOutputError
from .noddi import MATLABDiscoveryError, NODDIError, NODDIExternalCommandError
from .orchestrator import (
    STAGE_ORDER,
    PipelineDependencyError,
    PipelineExternalError,
    PipelineInputError,
    PipelineOutputError,
    StageSelection,
    run_pipeline,
)
from .preprocess import PreprocessError
from .qc import QCError
from .report import ReportError
from .resources import ResourceValidationError
from .state import StageStateError
from .stripe_qc import StripeQCError
from .summary import SummaryError


class _CLIError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="run_pipeline.sh",
        description="Run one patient-generic, resumable diffusion MRI analysis.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    bounds = parser.add_mutually_exclusive_group()
    bounds.add_argument("--stop-after", choices=STAGE_ORDER)
    bounds.add_argument("--only-stage", choices=STAGE_ORDER)
    parser.add_argument("--force-stage", choices=STAGE_ORDER)
    parser.add_argument("config", nargs=1, metavar="CONFIG.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, print concise errors, and return the exact exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        namespace = _parser().parse_args(arguments)
        if namespace.force_stage is not None and (
            namespace.validate_only or namespace.dry_run
        ):
            raise _CLIError("--force-stage cannot be combined with a nonmutating mode")
        mode = (
            "validate-only"
            if namespace.validate_only
            else "dry-run"
            if namespace.dry_run
            else "run"
        )
        selection = StageSelection(
            stop_after=namespace.stop_after,
            only_stage=namespace.only_stage,
        )
        if mode != "run" and selection != StageSelection():
            raise _CLIError("bounded execution is valid only for a normal run")
        if (
            namespace.force_stage is not None
            and namespace.force_stage not in selection.execution_names
        ):
            raise _CLIError(
                f"forced stage {namespace.force_stage} is outside the selected "
                "execution range"
            )
        config = load_config(Path(namespace.config[0]))
        outcome = run_pipeline(
            config,
            mode,
            namespace.force_stage,
            selection=selection,
        )
        print(
            f"RESULT subject={outcome.subject} status={outcome.status} "
            f"output={outcome.subject_output}"
        )
        return {
            "COMPLETE": 0,
            "PARTIAL_COMPLETE": 0,
            "STAGE_COMPLETE": 0,
            "VALIDATED": 0,
            "DRY_RUN": 0,
            "EXCLUDED": 20,
            "HOLD_FOR_REVIEW": 21,
        }.get(outcome.status, 50)
    except (_CLIError, ConfigError, InputAuditError, PipelineInputError) as error:
        return _error(2, error)
    except (
        FSLDiscoveryError,
        MATLABDiscoveryError,
        PipelineDependencyError,
        ResourceValidationError,
    ) as error:
        return _error(30, error)
    except (
        ExternalCommandError,
        NODDIExternalCommandError,
        PipelineExternalError,
    ) as error:
        return _error(40, error)
    except (
        PipelineOutputError,
        PreprocessError,
        NODDIError,
        StageStateError,
        StripeQCError,
        ModelInputError,
        ModelOutputError,
        SummaryError,
        QCError,
        ReportError,
        ValueError,
        TypeError,
        OSError,
    ) as error:
        return _error(50, error)


def _error(code: int, error: BaseException) -> int:
    message = str(error).strip() or type(error).__name__
    print(f"ERROR: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
