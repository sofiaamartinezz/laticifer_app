# utils/quantification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


from skimage.morphology import convex_hull_image, binary_closing, disk, binary_dilation, binary_erosion
from scipy import ndimage as ndi
from skimage.transform import rescale, resize

def generate_structure_based_tissue_mask(laticifer_mask: np.ndarray, method: str = 'envelope') -> np.ndarray:
    """
    Generates a tissue mask based solely on the location of laticifers.
    
    Args:
        laticifer_mask: Binary mask of laticifers.
        method: 'hull' (Convex Hull) or 'envelope' (Morphological closing/filling).
    """
    scale_factor = 0.25
    original_shape = laticifer_mask.shape
    
    small_mask = rescale(laticifer_mask, scale_factor, order=0, anti_aliasing=False).astype(bool)
    
    if not np.any(small_mask):
        return np.zeros(original_shape, dtype=np.uint8)

    if method == 'hull':
        tissue_small = convex_hull_image(small_mask)
        
    else:
        radius = 25 
        structure = disk(radius)
        dilated = binary_dilation(small_mask, structure)
        filled = ndi.binary_fill_holes(dilated)
        tissue_small = binary_erosion(filled, structure)

    tissue_mask = resize(tissue_small, original_shape, order=0, anti_aliasing=False)
    
    return tissue_mask.astype(np.uint8)


# ----------------------------
# pixel ratio
# ----------------------------
def analyze_density_pixel_ratio(
    mask: np.ndarray, 
    roi: Optional[Tuple[slice, slice]] = None,
    use_tissue_mask: bool = False
) -> Dict[str, object]:
    
    data = mask
    if roi is not None:
        ysl, xsl = roi
        data = data[ysl, xsl]

    laticifer_pixels = int(np.count_nonzero(data > 0))
    
    generated_mask_for_debug = None 

    if use_tissue_mask:
        tissue_mask = generate_structure_based_tissue_mask(data, method='hull')
        generated_mask_for_debug = tissue_mask 
        total_pixels = int(np.count_nonzero(tissue_mask))
        total_pixels = max(total_pixels, laticifer_pixels)
    else:
        total_pixels = int(data.size)

    if total_pixels == 0:
        ratio = 0.0
    else:
        ratio = laticifer_pixels / total_pixels

    return {
        "laticifer_pixels": laticifer_pixels,
        "total_pixels": total_pixels,
        "pixel_ratio": float(ratio),
        "density_percentage": float(ratio * 100.0),
        "debug_tissue_mask": generated_mask_for_debug
    }


# ==========================================================
# transect function  (original — generates lines internally)
# ==========================================================
def analyze_density_transect(
    mask: np.ndarray,
    num_lines: int = 10,
    direction: str = "both",
    roi: Optional[Tuple[slice, slice]] = None,
) -> Tuple[Dict[str, float], List[np.ndarray], np.ndarray]:
    """
    Calculate transect density with auto-generated lines.

    Lines are placed at pos_i = (i + 0.5) * L / n — uniformly distributed
    across the full axis with no border margin.
    """
    if mask.size == 0:
        return {"error": "Empty mask"}, [], np.zeros((0, 2), dtype=float)

    mask_bin = (mask > 0).astype(np.uint8)

    y_off = 0
    x_off = 0
    if roi is not None:
        ysl, xsl = roi
        y_off = ysl.start or 0
        x_off = xsl.start or 0
        mask_bin = mask_bin[ysl, xsl]

    H, W = mask_bin.shape[:2]
    if H == 0 or W == 0:
        return {"error": "Empty ROI"}, [], np.zeros((0, 2), dtype=float)

    direction = (direction or "both").lower().strip()
    if direction not in ("horizontal", "vertical", "both"):
        direction = "both"

    n = max(1, int(num_lines))

    lines: List[np.ndarray] = []
    points: List[Tuple[float, float]] = []

    horizontal_intersections: List[int] = []
    vertical_intersections: List[int] = []

    def _centered_positions(length: int) -> np.ndarray:
        """n equidistant positions covering the full axis, centred in each segment."""
        positions = (np.arange(n, dtype=float) + 0.5) * (float(length) / float(n))
        return np.clip(np.round(positions).astype(int), 0, max(0, length - 1))

    # --- HORIZONTAL ---
    if direction in ("horizontal", "both"):
        y_positions = _centered_positions(H)
        for y in y_positions:
            y = int(np.clip(y, 0, H - 1))

            y_g = float(y + y_off)
            lines.append(np.array([[y_g, float(0 + x_off)], [y_g, float((W - 1) + x_off)]], dtype=float))

            line_pixels = mask_bin[y, :]
            diffs = np.diff(line_pixels.astype(np.int8))

            num_intersections = int(np.count_nonzero(diffs == 1))
            horizontal_intersections.append(num_intersections)

            xs = np.where(diffs == 1)[0] + 1
            for x in xs:
                points.append((float(y + y_off), float(x + x_off)))

    # --- VERTICAL ---
    if direction in ("vertical", "both"):
        x_positions = _centered_positions(W)
        for x in x_positions:
            x = int(np.clip(x, 0, W - 1))

            x_g = float(x + x_off)
            lines.append(np.array([[float(0 + y_off), x_g], [float((H - 1) + y_off), x_g]], dtype=float))

            line_pixels = mask_bin[:, x]
            diffs = np.diff(line_pixels.astype(np.int8))

            num_intersections = int(np.count_nonzero(diffs == 1))
            vertical_intersections.append(num_intersections)

            ys = np.where(diffs == 1)[0] + 1
            for y in ys:
                points.append((float(y + y_off), float(x + x_off)))

    all_intersections = horizontal_intersections + vertical_intersections

    stats: Dict[str, float] = {
        "num_lines": float(n),
        "direction": direction,
    }

    if all_intersections:
        stats["mean_intersections_per_line"] = float(np.mean(all_intersections))
        stats["std_intersections_per_line"] = float(np.std(all_intersections))
    else:
        stats["mean_intersections_per_line"] = float("nan")
        stats["std_intersections_per_line"] = float("nan")

    if horizontal_intersections:
        stats["mean_horizontal_intersections"] = float(np.mean(horizontal_intersections))
    else:
        stats["mean_horizontal_intersections"] = float("nan")

    if vertical_intersections:
        stats["mean_vertical_intersections"] = float(np.mean(vertical_intersections))
    else:
        stats["mean_vertical_intersections"] = float("nan")

    pts = np.array(points, dtype=float) if points else np.zeros((0, 2), dtype=float)
    return stats, lines, pts


# ==========================================================
# NEW: recalculate density using existing line geometry
# ==========================================================
def analyze_density_from_lines(
    mask: np.ndarray,
    horizontal_lines: List[np.ndarray],
    vertical_lines: List[np.ndarray],
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calculate transect density using the *current* geometry of provided lines.

    This is used when the user has moved or deleted transects manually and
    clicks "Recalcular densidad" — we must NOT regenerate positions.

    Args:
        mask:             Raw label mask (binarised internally as > 0).
        horizontal_lines: List of (2,2) arrays [[y, x0], [y, x1]] (y0 == y1).
        vertical_lines:   List of (2,2) arrays [[y0, x], [y1, x]] (x0 == x1).

    Returns:
        stats:  Dict with mean/std intersections (same keys as analyze_density_transect).
        pts:    (N, 2) float array of entry points (y, x) for napari Points layer.

    Notes:
        - Lines are expected in napari global coordinates (no ROI offset here).
        - Each horizontal line is sampled at its rounded Y coordinate.
        - Each vertical line is sampled at its rounded X coordinate.
        - Intersection count: transitions 0→1 in the pixel profile (diffs == 1).
    """
    if mask.size == 0:
        empty_stats = {
            "num_lines": 0.0,
            "direction": "mixed",
            "mean_intersections_per_line": float("nan"),
            "std_intersections_per_line": float("nan"),
            "mean_horizontal_intersections": float("nan"),
            "mean_vertical_intersections": float("nan"),
        }
        return empty_stats, np.zeros((0, 2), dtype=float)

    mask_bin = (mask > 0).astype(np.uint8)
    H, W = mask_bin.shape[:2]

    points: List[Tuple[float, float]] = []
    horizontal_intersections: List[int] = []
    vertical_intersections: List[int] = []

    # --- Process horizontal lines ---
    for line_coords in horizontal_lines:
        # line_coords: [[y, x0], [y, x1]]  (y0 == y1 for horizontal)
        y = int(round(float(line_coords[0, 0])))
        y = int(np.clip(y, 0, H - 1))

        line_pixels = mask_bin[y, :]
        diffs = np.diff(line_pixels.astype(np.int8))
        num_intersections = int(np.count_nonzero(diffs == 1))
        horizontal_intersections.append(num_intersections)

        xs = np.where(diffs == 1)[0] + 1
        for x in xs:
            points.append((float(y), float(x)))

    # --- Process vertical lines ---
    for line_coords in vertical_lines:
        # line_coords: [[y0, x], [y1, x]]  (x0 == x1 for vertical)
        x = int(round(float(line_coords[0, 1])))
        x = int(np.clip(x, 0, W - 1))

        line_pixels = mask_bin[:, x]
        diffs = np.diff(line_pixels.astype(np.int8))
        num_intersections = int(np.count_nonzero(diffs == 1))
        vertical_intersections.append(num_intersections)

        ys = np.where(diffs == 1)[0] + 1
        for y in ys:
            points.append((float(y), float(x)))

    all_intersections = horizontal_intersections + vertical_intersections
    total_lines = len(all_intersections)

    # Determine direction label for stats
    has_h = len(horizontal_intersections) > 0
    has_v = len(vertical_intersections) > 0
    if has_h and has_v:
        direction_label = "both"
    elif has_h:
        direction_label = "horizontal"
    elif has_v:
        direction_label = "vertical"
    else:
        direction_label = "none"

    stats: Dict[str, float] = {
        "num_lines": float(total_lines),
        "direction": direction_label,
    }

    if all_intersections:
        stats["mean_intersections_per_line"] = float(np.mean(all_intersections))
        stats["std_intersections_per_line"] = float(np.std(all_intersections))
    else:
        stats["mean_intersections_per_line"] = float("nan")
        stats["std_intersections_per_line"] = float("nan")

    if horizontal_intersections:
        stats["mean_horizontal_intersections"] = float(np.mean(horizontal_intersections))
    else:
        stats["mean_horizontal_intersections"] = float("nan")

    if vertical_intersections:
        stats["mean_vertical_intersections"] = float(np.mean(vertical_intersections))
    else:
        stats["mean_vertical_intersections"] = float("nan")

    pts = np.array(points, dtype=float) if points else np.zeros((0, 2), dtype=float)
    return stats, pts