"""Strict EDDY command and stage runtime evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


_KEYS = frozenset(
    {
        "schema_version",
        "eddy_command_seconds",
        "eddy_quad_seconds",
        "stage_action_seconds",
        "eddy_command_includes_cnr_maps",
        "eddy_command_includes_residuals",
    }
)


class EddyTimingError(ValueError):
    """EDDY timing JSON is missing, unreadable, or scientifically ambiguous."""


@dataclass(frozen=True)
class EddyTiming:
    """Validated monotonic durations for the complete EDDY stage action."""

    eddy_command_seconds: float
    eddy_quad_seconds: float
    stage_action_seconds: float


def parse_eddy_timing(payload: object) -> EddyTiming:
    """Parse the exact version-1 timing schema into an immutable value."""
    if not isinstance(payload, Mapping) or set(payload) != _KEYS:
        raise EddyTimingError("EDDY timing must contain the exact version-1 key set")
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise EddyTimingError("EDDY timing schema_version must be integer 1")
    for key in (
        "eddy_command_includes_cnr_maps",
        "eddy_command_includes_residuals",
    ):
        if payload[key] is not True:
            raise EddyTimingError(f"EDDY timing {key} must be true")

    values: dict[str, float] = {}
    for key in (
        "eddy_command_seconds",
        "eddy_quad_seconds",
        "stage_action_seconds",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EddyTimingError(f"EDDY timing {key} must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise EddyTimingError(f"EDDY timing {key} must be finite and nonnegative")
        values[key] = normalized

    timing = EddyTiming(**values)
    component_total = timing.eddy_command_seconds + timing.eddy_quad_seconds
    if timing.stage_action_seconds < component_total:
        raise EddyTimingError("EDDY stage total cannot be smaller than its commands")
    return timing


def read_eddy_timing(path: Path) -> EddyTiming:
    """Read and strictly validate one EDDY timing JSON file."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EddyTimingError("cannot read EDDY timing JSON") from error
    return parse_eddy_timing(payload)


def write_eddy_timing(path: Path, timing: EddyTiming) -> None:
    """Write deterministic version-1 timing evidence after strict validation."""
    if not isinstance(timing, EddyTiming):
        raise TypeError("timing must be EddyTiming")
    payload = {
        "schema_version": 1,
        "eddy_command_seconds": timing.eddy_command_seconds,
        "eddy_quad_seconds": timing.eddy_quad_seconds,
        "stage_action_seconds": timing.stage_action_seconds,
        "eddy_command_includes_cnr_maps": True,
        "eddy_command_includes_residuals": True,
    }
    parse_eddy_timing(payload)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
