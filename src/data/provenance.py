"""Shared provenance values written with every exported analysis."""
from __future__ import annotations

from datetime import datetime


APP_VERSION = "0.1.0-dev"


def analysis_timestamp() -> str:
    """Return a timezone-aware ISO 8601 timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def measurement_units(um_per_px: float | None) -> tuple[str, str, str]:
    """Describe whether exported distances use physical or pixel units."""
    if um_per_px is not None and um_per_px > 0:
        return "physical", "µm", "µm²"
    return "pixels_only", "px", "px²"
