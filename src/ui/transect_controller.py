# ui/transect_controller.py
"""
TransectController: manages editable transect lines in a napari viewer.

Design:
- generate():     computes centered positions and creates the Shapes layer.
                  Listens to layer.events.data to detect user deletions → pending flag.
- recalculate():  reads whatever lines exist in the Shapes layer right now,
                  adds intersection points below the shapes layer, re-selects shapes.
- pending flag:   True after generate or user edits, False after recalculate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import napari
from napari.layers import Shapes

from utils.quantification import analyze_density_from_lines


TRANSECT_LAYER_NAME = "Transect lines"
POINTS_LAYER_NAME   = "Intersection points"


# ---------------------------------------------------------------------------
#  Position generation
# ---------------------------------------------------------------------------

@dataclass
class _TransectLine:
    direction: str       # 'horizontal' | 'vertical'
    coords: np.ndarray   # (2, 2) float [[y0, x0], [y1, x1]]

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        assert self.coords.shape == (2, 2), "coords must be (2, 2)"


def _generate_lines(
    image_shape: Tuple[int, int],
    num_lines: int,
    direction: str,
) -> List[np.ndarray]:
    """
    Return (2, 2) coordinate arrays for centered, equidistant transects.

    Formula: pos_i = (i + 0.5) * L / n
    Lines cover the full axis uniformly with no border margin.
    """
    H, W      = image_shape
    n         = max(1, int(num_lines))
    direction = (direction or "both").lower().strip()
    lines: List[_TransectLine] = []

    def _positions(length: int) -> np.ndarray:
        pos = (np.arange(n, dtype=float) + 0.5) * (float(length) / float(n))
        return np.clip(np.round(pos).astype(int), 0, max(0, length - 1))

    if direction in ("horizontal", "both"):
        for y in _positions(H):
            lines.append(_TransectLine(
                direction="horizontal",
                coords=np.array([[float(y), 0.0], [float(y), float(W - 1)]]),
            ))

    if direction in ("vertical", "both"):
        for x in _positions(W):
            lines.append(_TransectLine(
                direction="vertical",
                coords=np.array([[0.0, float(x)], [float(H - 1), float(x)]]),
            ))

    return [l.coords.copy() for l in lines]


# ---------------------------------------------------------------------------
#  Controller
# ---------------------------------------------------------------------------

class TransectController:

    def __init__(
        self,
        viewer: napari.Viewer,
        on_state_change: Optional[Callable[[], None]] = None,
    ):
        self.viewer          = viewer
        self._on_state_change = on_state_change or (lambda: None)
        self.last_stats: Optional[dict] = None
        self._shapes_layer: Optional[Shapes] = None
        self._pending: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        mask_shape: Tuple[int, int],
        num_lines: int,
        direction: str,
    ) -> None:
        """
        Generate centered transect lines and render them in a new Shapes layer.
        Replaces any existing Transect lines and Intersection points layers.
        """
        self._remove_layer(TRANSECT_LAYER_NAME)
        self._remove_layer(POINTS_LAYER_NAME)

        napari_lines = _generate_lines(mask_shape, num_lines, direction)
        if not napari_lines:
            self._shapes_layer = None
            self._pending      = False
            self._on_state_change()
            return

        self._shapes_layer = self.viewer.add_shapes(
            napari_lines,
            name=TRANSECT_LAYER_NAME,
            shape_type="line",
            edge_width=8,
            edge_color="cyan",
        )
        self._shapes_layer.mode = "select"
        self.viewer.layers.selection.active = self._shapes_layer
        self._shapes_layer.events.data.connect(self._on_shapes_data_changed)

        self._pending = True
        self._on_state_change()

    def recalculate(
        self,
        mask: np.ndarray,
        show_points: bool = True,
        um_per_px: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Recalculate density from whatever lines currently exist in the Shapes layer.
        Lines are NOT regenerated — user deletions are preserved.

        Args:
            mask:        Current binary label mask.
            show_points: Add a napari Points layer with intersection markers.
            um_per_px:   Optional scale factor forwarded to the analysis function.

        Returns:
            Stats dict (same keys as analyze_density_from_lines) or None on failure.
        """
        self._remove_layer(POINTS_LAYER_NAME)

        lines = self._read_lines_from_layer()
        if not lines:
            return None

        h_lines, v_lines = self._classify_lines(lines)
        stats, pts       = analyze_density_from_lines(
            mask, h_lines, v_lines, um_per_px=um_per_px
        )
        self.last_stats = stats

        if show_points and pts.shape[0] > 0:
            self.viewer.add_points(
                pts, name=POINTS_LAYER_NAME, size=6, face_color="yellow"
            )
            # Keep shapes on top so the user can keep editing
            shapes = self._find_shapes_layer()
            if shapes is not None:
                idx = self.viewer.layers.index(shapes)
                top = len(self.viewer.layers) - 1
                if idx != top:
                    self.viewer.layers.move(idx, top)
                self.viewer.layers.selection.active = shapes

        self._pending = False
        self._on_state_change()
        return stats

    @property
    def pending(self) -> bool:
        return self._pending

    @property
    def has_transects(self) -> bool:
        return len(self._read_lines_from_layer()) > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_shapes_data_changed(self, event=None) -> None:
        """Fired by napari when the user deletes or adds a shape."""
        self._pending = True
        self._on_state_change()

    def _read_lines_from_layer(self) -> List[np.ndarray]:
        layer = self._find_shapes_layer()
        if layer is None:
            return []
        result = []
        for coords in layer.data:
            arr = np.asarray(coords, dtype=float)
            if arr.shape == (2, 2):
                result.append(arr)
            elif arr.ndim == 2 and arr.shape[0] >= 2:
                result.append(arr[[0, -1], :])
        return result

    def _find_shapes_layer(self) -> Optional[Shapes]:
        if TRANSECT_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[TRANSECT_LAYER_NAME]
            if isinstance(layer, Shapes):
                return layer
        return None

    @staticmethod
    def _classify_lines(
        lines: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split lines into horizontal / vertical by comparing dy vs dx."""
        h_lines, v_lines = [], []
        for coords in lines:
            dy = abs(float(coords[1, 0]) - float(coords[0, 0]))
            dx = abs(float(coords[1, 1]) - float(coords[0, 1]))
            (h_lines if dx >= dy else v_lines).append(coords)
        return h_lines, v_lines

    def _remove_layer(self, name: str) -> None:
        if name in self.viewer.layers:
            try:
                self.viewer.layers.remove(name)
            except Exception:
                pass