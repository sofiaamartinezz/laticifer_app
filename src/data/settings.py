"""Typed, validated, persistent application preferences."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from qtpy.QtCore import QSettings


@dataclass(frozen=True)
class AppSettings:
    transect_num_lines: int = 10
    remove_small_min_size: int = 100
    fill_holes_area: int = 500
    morphology_radius: int = 1
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    minimum_reference_line_px: float = 10.0
    minimum_typical_um_per_px: float = 0.001
    maximum_typical_um_per_px: float = 100.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AppSettings":
        defaults = cls()
        specs = {
            "transect_num_lines": (int, 1, 10_000),
            "remove_small_min_size": (int, 1, 100_000),
            "fill_holes_area": (int, 1, 1_000_000),
            "morphology_radius": (int, 1, 100),
            "clahe_clip_limit": (float, 0.1, 100.0),
            "clahe_tile_size": (int, 1, 128),
            "minimum_reference_line_px": (float, 1.0, 10_000.0),
            "minimum_typical_um_per_px": (float, 1e-9, 1_000_000.0),
            "maximum_typical_um_per_px": (float, 1e-9, 1_000_000.0),
        }
        parsed = {}
        for name, (converter, minimum, maximum) in specs.items():
            raw = values.get(name, getattr(defaults, name))
            try:
                value = converter(raw)
                if not minimum <= value <= maximum:
                    raise ValueError
            except (TypeError, ValueError):
                value = getattr(defaults, name)
            parsed[name] = value
        if parsed["minimum_typical_um_per_px"] >= parsed["maximum_typical_um_per_px"]:
            parsed["minimum_typical_um_per_px"] = defaults.minimum_typical_um_per_px
            parsed["maximum_typical_um_per_px"] = defaults.maximum_typical_um_per_px
        return cls(**parsed)

    def updated(self, **values: Any) -> "AppSettings":
        return self.from_mapping(asdict(replace(self, **values)))


class SettingsStore:
    """Persist preferences in the platform's standard Qt user-settings area."""

    PREFIX = "preferences"

    def __init__(self, backend=None) -> None:
        self._backend = backend or QSettings("LatexLens", "LatexLens")

    def load(self) -> AppSettings:
        defaults = asdict(AppSettings())
        values = {
            key: self._backend.value(f"{self.PREFIX}/{key}", default)
            for key, default in defaults.items()
        }
        return AppSettings.from_mapping(values)

    def save(self, settings: AppSettings) -> None:
        for key, value in asdict(settings).items():
            self._backend.setValue(f"{self.PREFIX}/{key}", value)
        if hasattr(self._backend, "sync"):
            self._backend.sync()

    def reset(self) -> AppSettings:
        self._backend.remove(self.PREFIX)
        if hasattr(self._backend, "sync"):
            self._backend.sync()
        return AppSettings()
