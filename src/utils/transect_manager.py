# utils/transect_manager.py
"""
Manages transect state: generation, movement constraints, deletion.

Responsibilities:
- Store current transect lines as (2,2) napari-compatible arrays
- Generate centered, equidistant positions
- Enforce axis-constrained movement (horizontal → only vertical, vertical → only horizontal)
- Track which index is selected
- Flag when transects have been manually edited (pending recalculation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# -----------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------

@dataclass
class TransectLine:
    """
    A single transect line stored in napari coordinate space (y, x).

    direction: 'horizontal' or 'vertical'
    coords:    (2, 2) array [[y0, x0], [y1, x1]]
                - horizontal: y0 == y1, x spans full image width
                - vertical:   x0 == x1, y spans full image height
    """
    direction: str           # 'horizontal' | 'vertical'
    coords: np.ndarray       # shape (2, 2), dtype float

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        assert self.coords.shape == (2, 2), "coords must be (2, 2)"


# -----------------------------------------------------------------------
# Manager
# -----------------------------------------------------------------------

class TransectManager:
    """
    Central controller for transect lines in the interactive editor.

    Usage:
        manager = TransectManager()
        manager.generate(mask_shape, num_lines=5, direction="horizontal")
        manager.move(index=0, delta=-10)      # move line 0 up by 10 px
        manager.delete(index=1)               # remove line 1
        lines = manager.get_napari_lines()    # list of (2,2) arrays for Shapes layer
    """

    def __init__(self):
        self._lines: List[TransectLine] = []
        self._selected_index: Optional[int] = None
        self._pending: bool = False            # True after any edit since last recalc
        self._image_shape: Optional[Tuple[int, int]] = None  # (H, W)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_recalculation(self) -> bool:
        """True when transects were edited since the last density calculation."""
        return self._pending

    @property
    def selected_index(self) -> Optional[int]:
        return self._selected_index

    @property
    def count(self) -> int:
        return len(self._lines)

    @property
    def image_shape(self) -> Optional[Tuple[int, int]]:
        return self._image_shape

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image_shape: Tuple[int, int],
        num_lines: int,
        direction: str,
    ) -> None:
        """
        Generate centered, equidistant transect lines covering the full image.

        Uses pos_i = (i + 0.5) * L / n  so lines are evenly spaced across the
        entire axis with no border margin, but without touching the edges.
        """
        self._lines = []
        self._selected_index = None
        self._image_shape = image_shape

        H, W = image_shape
        n = max(1, int(num_lines))
        direction = (direction or "both").lower().strip()

        def _positions(length: int) -> np.ndarray:
            """n equidistant positions covering the full axis, centred in each segment."""
            positions = (np.arange(n, dtype=float) + 0.5) * (float(length) / float(n))
            return np.clip(np.round(positions).astype(int), 0, max(0, length - 1))

        if direction in ("horizontal", "both"):
            for y in _positions(H):
                coords = np.array([[float(y), 0.0], [float(y), float(W - 1)]], dtype=float)
                self._lines.append(TransectLine(direction="horizontal", coords=coords))

        if direction in ("vertical", "both"):
            for x in _positions(W):
                coords = np.array([[0.0, float(x)], [float(H - 1), float(x)]], dtype=float)
                self._lines.append(TransectLine(direction="vertical", coords=coords))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, index: Optional[int]) -> None:
        """Set the currently selected transect index (None = deselect)."""
        if index is None or 0 <= index < len(self._lines):
            self._selected_index = index

    def select_by_napari_index(self, napari_index: int) -> None:
        """
        napari Shapes layer uses its own ordering; map directly by list index.
        The Shapes layer order matches self._lines order after each sync.
        """
        self.select(napari_index if 0 <= napari_index < len(self._lines) else None)

    # ------------------------------------------------------------------
    # Movement  (axis-constrained)
    # ------------------------------------------------------------------

    def move(self, index: int, delta: float) -> bool:
        """
        Move a transect line by `delta` pixels along its constrained axis.

        - horizontal line → delta applied to Y coordinate (vertical movement)
        - vertical line   → delta applied to X coordinate (horizontal movement)

        The line is clamped to image bounds if image_shape is known.

        Returns True on success, False if index is invalid.
        """
        if not (0 <= index < len(self._lines)):
            return False

        line = self._lines[index]
        H, W = self._image_shape if self._image_shape else (None, None)

        if line.direction == "horizontal":
            # Move vertically: change Y for both endpoints
            new_y = line.coords[0, 0] + delta
            if H is not None:
                new_y = float(np.clip(new_y, 0, H - 1))
            line.coords[:, 0] = new_y          # set y0 and y1

        elif line.direction == "vertical":
            # Move horizontally: change X for both endpoints
            new_x = line.coords[0, 1] + delta
            if W is not None:
                new_x = float(np.clip(new_x, 0, W - 1))
            line.coords[:, 1] = new_x          # set x0 and x1

        self._pending = True
        return True

    def move_selected(self, delta: float) -> bool:
        """Move the currently selected transect. Returns False if none selected."""
        if self._selected_index is None:
            return False
        return self.move(self._selected_index, delta)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, index: int) -> bool:
        """
        Delete the transect at `index`.

        Returns True on success.
        """
        if not (0 <= index < len(self._lines)):
            return False
        self._lines.pop(index)
        # Adjust selected index
        if self._selected_index is not None:
            if self._selected_index == index:
                self._selected_index = None
            elif self._selected_index > index:
                self._selected_index -= 1
        self._pending = True
        return True

    def delete_selected(self) -> bool:
        """Delete the currently selected transect. Returns False if none selected."""
        if self._selected_index is None:
            return False
        return self.delete(self._selected_index)

    # ------------------------------------------------------------------
    # Mark recalculated
    # ------------------------------------------------------------------

    def mark_recalculated(self) -> None:
        """Call after a successful density recalculation to clear the pending flag."""
        self._pending = False

    # ------------------------------------------------------------------
    # Napari output
    # ------------------------------------------------------------------

    def get_napari_lines(self) -> list[np.ndarray]:
        """
        Return a list of (2,2) arrays suitable for napari's Shapes layer.
        Each array: [[y0, x0], [y1, x1]]
        """
        return [line.coords.copy() for line in self._lines]

    def get_lines_by_direction(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return (horizontal_lines, vertical_lines) as separate lists of (2,2) arrays.
        Used by the recalculation logic in quantification.py.
        """
        h = [l.coords.copy() for l in self._lines if l.direction == "horizontal"]
        v = [l.coords.copy() for l in self._lines if l.direction == "vertical"]
        return h, v

    def is_empty(self) -> bool:
        return len(self._lines) == 0