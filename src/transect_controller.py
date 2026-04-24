# ui/transect_controller.py
"""
TransectController: bridges napari viewer with transect logic.

Design:
- generate(): creates the Shapes layer (lines on top, no points yet).
              Listens to layer.events.data to detect user deletions → pending flag.
- recalculate(): reads whatever lines exist in the Shapes layer right now,
                 adds points BELOW the shapes layer, re-selects shapes.
- pending flag: False after generate/recalculate, True only after user edits.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np

import napari
from napari.layers import Shapes, Points

from utils.transect_manager import TransectManager
from utils.quantification import analyze_density_from_lines


TRANSECT_LAYER_NAME = "Transect lines"
POINTS_LAYER_NAME = "Intersection points"


class TransectController:

    def __init__(
        self,
        viewer: napari.Viewer,
        on_state_change: Optional[Callable[[], None]] = None,
    ):
        self.viewer = viewer
        self._on_state_change = on_state_change or (lambda: None)
        self.last_stats: Optional[dict] = None
        self._shapes_layer: Optional[Shapes] = None
        self._manager = TransectManager()
        self._pending: bool = False  # True only after user edits lines

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
        Generate centered, equidistant transect lines and render them.
        Replaces existing Transect lines and Intersection points layers.
        pending is reset to False (no edits yet).
        """
        self._remove_layer(TRANSECT_LAYER_NAME)
        self._remove_layer(POINTS_LAYER_NAME)
        self._pending = False

        self._manager.generate(
            image_shape=mask_shape,
            num_lines=num_lines,
            direction=direction,
        )
        napari_lines = self._manager.get_napari_lines()

        if not napari_lines:
            self._shapes_layer = None
            self._on_state_change()
            return

        # Render shapes on top (added last = top of stack).
        # edge_width=8 so lines are easy to click/select.
        self._shapes_layer = self.viewer.add_shapes(
            napari_lines,
            name=TRANSECT_LAYER_NAME,
            shape_type="line",
            edge_width=8,
            edge_color="cyan",
        )
        self._shapes_layer.mode = "select"
        self.viewer.layers.selection.active = self._shapes_layer

        # Detect user deletions/additions via the data event → show pending warning
        self._shapes_layer.events.data.connect(self._on_shapes_data_changed)

        self._on_state_change()

    def _on_shapes_data_changed(self, event=None) -> None:
        """Called by napari when the user deletes or adds a shape. Sets pending."""
        self._pending = True
        self._on_state_change()

    def recalculate(self, mask: np.ndarray, show_points: bool = True) -> Optional[dict]:
        """
        Recalculate density from whatever lines currently exist in the Shapes layer.
        Points are added below shapes; shapes layer is re-selected after.
        pending is reset to False after a successful recalculate.
        """
        # Remove old points before re-adding (so they go below shapes in the stack)
        self._remove_layer(POINTS_LAYER_NAME)

        lines = self._read_lines_from_layer()
        if not lines:
            return None

        h_lines, v_lines = self._classify_lines(lines)
        stats, pts = analyze_density_from_lines(mask, h_lines, v_lines)
        self.last_stats = stats

        if show_points and pts.shape[0] > 0:
            # 1. Add points — they land on top of the stack momentarily
            self.viewer.add_points(pts, name=POINTS_LAYER_NAME, size=6, face_color="yellow")

            # 2. Move shapes layer above points so transects stay on top
            shapes = self._find_shapes_layer()
            if shapes is not None:
                current_idx = self.viewer.layers.index(shapes)
                top_idx = len(self.viewer.layers) - 1
                if current_idx != top_idx:
                    self.viewer.layers.move(current_idx, top_idx)
                # Re-activate shapes so user can keep editing
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
    # Reading lines from napari layer
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Classifying lines
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_lines(
        lines: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        h_lines: List[np.ndarray] = []
        v_lines: List[np.ndarray] = []
        for coords in lines:
            dy = abs(float(coords[1, 0]) - float(coords[0, 0]))
            dx = abs(float(coords[1, 1]) - float(coords[0, 1]))
            if dx >= dy:
                h_lines.append(coords)
            else:
                v_lines.append(coords)
        return h_lines, v_lines

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _remove_layer(self, name: str) -> None:
        if name in self.viewer.layers:
            try:
                self.viewer.layers.remove(name)
            except Exception:
                pass