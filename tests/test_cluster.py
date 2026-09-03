from __future__ import annotations

from dataclasses import replace

import pytest

from dmri_pipeline.cluster import ClusterConfigError, cluster_subject_context


def test_cluster_context_uses_validated_subject_values(subject_config) -> None:
    """Configured workers are preserved in the nonmutating cluster context."""
    configured = replace(
        subject_config,
        analysis=replace(subject_config.analysis, noddi_workers=3),
    )

    context = cluster_subject_context(configured)

    assert context.subject_id == "SYNTH001"
    assert context.subject_output == configured.output_root / "SYNTH001"
    assert context.noddi_workers == 3
    assert not context.subject_output.exists()


def test_cluster_context_rejects_automatic_noddi_workers(subject_config) -> None:
    """Cluster wrappers must receive an explicit worker count."""
    automatic = replace(
        subject_config,
        analysis=replace(subject_config.analysis, noddi_workers="auto"),
    )

    with pytest.raises(ClusterConfigError, match="explicit"):
        cluster_subject_context(automatic)
