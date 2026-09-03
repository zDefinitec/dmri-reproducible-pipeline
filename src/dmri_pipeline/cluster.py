"""Validated, nonmutating context for cluster-stage wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig


class ClusterConfigError(ValueError):
    """Raised when configuration cannot safely drive a cluster wrapper."""


@dataclass(frozen=True)
class ClusterSubjectContext:
    """The subject values required before launching cluster stages."""

    subject_id: str
    subject_output: Path
    noddi_workers: int


def cluster_subject_context(config: PipelineConfig) -> ClusterSubjectContext:
    """Return explicit, validated cluster values without mutating the filesystem."""
    workers = config.analysis.noddi_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ClusterConfigError(
            "cluster execution requires an explicit positive noddi_workers value"
        )
    return ClusterSubjectContext(
        subject_id=config.subject_id,
        subject_output=config.subject_output,
        noddi_workers=workers,
    )
