# utils/quantification.py
"""
Density quantification for laticifer masks.

Two strategies:
  1. Pixel-ratio  – fraction of pixels occupied by laticifers.
  2. Transect     – mean number of laticifer boundary crossings per scan line.

Both functions accept an optional `um_per_px` argument.  When provided,
real-unit values are added to the returned dict under keys suffixed `_um`
(or `_um2` for areas).  When absent (None), only pixel-based values are
returned and the `_um` keys are set to None.

Pure numpy / scipy / skimage — no Qt, no napari.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as raster_line
from skimage.morphology import convex_hull_image, disk, binary_dilation, binary_erosion
from skimage.transform import rescale, resize


# ---------------------------------------------------------------------------
#  Tissue mask generation
# ---------------------------------------------------------------------------

def generate_structure_based_tissue_mask(
    laticifer_mask: np.ndarray,
    method: str = "envelope",
) -> np.ndarray:
    """
    Estimate the tissue area from the spatial distribution of laticifers.

    Args:
        laticifer_mask: Binary mask of laticifers.
        method:         'hull' (convex hull) or 'envelope' (morphological shrink-wrap).

    Returns:
        Binary uint8 mask of the estimated tissue region.
    """
    original_shape = laticifer_mask.shape
    # Downscale to 25 % for speed on large images.
    small_mask = rescale(laticifer_mask, 0.25, order=0, anti_aliasing=False).astype(bool)

    if not np.any(small_mask):
        return np.zeros(original_shape, dtype=np.uint8)

    if method == "hull":
        tissue_small = convex_hull_image(small_mask)
    else:
        structure = disk(25)
        tissue_small = binary_erosion(
            ndi.binary_fill_holes(binary_dilation(small_mask, structure)),
            structure,
        )

    return resize(tissue_small, original_shape, order=0, anti_aliasing=False).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Pixel-ratio density
# ---------------------------------------------------------------------------

def analyze_density_pixel_ratio(
    mask: np.ndarray,
    roi: Optional[Tuple[slice, slice]] = None,
    use_tissue_mask: bool = False,
    um_per_px: Optional[float] = None,
) -> Dict:
    """
    Compute laticifer density as a pixel fraction.

    Args:
        mask:            Label mask (values > 0 are laticifers).
        roi:             Optional (y_slice, x_slice) crop.
        use_tissue_mask: If True, denominator is the auto-detected tissue area.
        um_per_px:       Pixel → micron scale factor.  When set, adds real-unit
                         keys to the returned dict.

    Returns:
        Dict with keys:
            laticifer_pixels, total_pixels, pixel_ratio, density_percentage,
            debug_tissue_mask,
            laticifer_area_um2  (None if um_per_px is None),
            total_area_um2      (None if um_per_px is None).
    """
    data = mask if roi is None else mask[roi[0], roi[1]]
    laticifer_pixels = int(np.count_nonzero(data > 0))
    debug_tissue_mask = None

    if use_tissue_mask:
        tissue_mask       = generate_structure_based_tissue_mask(data, method="hull")
        debug_tissue_mask = tissue_mask
        total_pixels      = max(int(np.count_nonzero(tissue_mask)), laticifer_pixels)
    else:
        total_pixels = int(data.size)

    ratio = laticifer_pixels / total_pixels if total_pixels > 0 else 0.0

    # Real-unit areas
    laticifer_area_um2: Optional[float] = None
    total_area_um2:     Optional[float] = None
    if um_per_px is not None and um_per_px > 0:
        px2_to_um2         = um_per_px ** 2
        laticifer_area_um2 = laticifer_pixels * px2_to_um2
        total_area_um2     = total_pixels     * px2_to_um2

    return {
        "laticifer_pixels":    laticifer_pixels,
        "total_pixels":        total_pixels,
        "pixel_ratio":         float(ratio),
        "density_percentage":  float(ratio * 100.0),
        "debug_tissue_mask":   debug_tissue_mask,
        "laticifer_area_um2":  laticifer_area_um2,
        "total_area_um2":      total_area_um2,
    }


# ---------------------------------------------------------------------------
#  Transect density — auto-generated lines
# ---------------------------------------------------------------------------

def _count_entries(values: np.ndarray) -> Tuple[int, np.ndarray]:
    """Count foreground runs, including one that starts at the line boundary."""
    values = np.asarray(values, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0, np.zeros(0, dtype=int)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    return int(starts.size), starts


def analyze_density_transect(
    mask: np.ndarray,
    num_lines: int = 10,
    direction: str = "both",
    roi: Optional[Tuple[slice, slice]] = None,
    um_per_px: Optional[float] = None,
) -> Tuple[Dict, List[np.ndarray], np.ndarray]:
    """
    Compute transect density with auto-generated, uniformly distributed lines.

    Position formula: pos_i = (i + 0.5) * L / n
    Lines cover the full axis with no border margin.

    Args:
        mask:      Label mask (binarised internally as > 0).
        num_lines: Number of lines per direction.
        direction: 'horizontal', 'vertical', or 'both'.
        roi:       Optional (y_slice, x_slice) crop.
        um_per_px: Pixel → micron scale factor.  When set, adds
                   `mean_intersections_per_um` to the stats dict.

    Returns:
        stats: Dict with mean/std intersections per line and per direction.
        lines: List of (2, 2) arrays for a napari Shapes layer.
        pts:   (N, 2) float array of intersection entry points (y, x).
    """
    if mask.size == 0:
        return {"error": "Empty mask"}, [], np.zeros((0, 2), dtype=float)

    mask_bin = (mask > 0).astype(np.uint8)

    y_off, x_off = 0, 0
    if roi is not None:
        ysl, xsl = roi
        y_off    = ysl.start or 0
        x_off    = xsl.start or 0
        mask_bin = mask_bin[ysl, xsl]

    H, W = mask_bin.shape[:2]
    if H == 0 or W == 0:
        return {"error": "Empty ROI"}, [], np.zeros((0, 2), dtype=float)

    direction = (direction or "both").lower().strip()
    if direction not in ("horizontal", "vertical", "both"):
        direction = "both"

    n = max(1, int(num_lines))
    lines:    List[np.ndarray] = []
    points:   List[Tuple[float, float]] = []
    h_counts: List[int] = []
    v_counts: List[int] = []

    def _positions(length: int) -> np.ndarray:
        pos = (np.arange(n, dtype=float) + 0.5) * (float(length) / float(n))
        return np.clip(np.round(pos).astype(int), 0, max(0, length - 1))

    if direction in ("horizontal", "both"):
        for y in _positions(H):
            y   = int(y)
            y_g = float(y + y_off)
            lines.append(np.array([[y_g, float(x_off)], [y_g, float(W - 1 + x_off)]], dtype=float))
            count, starts = _count_entries(mask_bin[y, :])
            h_counts.append(count)
            for x in starts:
                points.append((float(y + y_off), float(x + x_off)))

    if direction in ("vertical", "both"):
        for x in _positions(W):
            x   = int(x)
            x_g = float(x + x_off)
            lines.append(np.array([[float(y_off), x_g], [float(H - 1 + y_off), x_g]], dtype=float))
            count, starts = _count_entries(mask_bin[:, x])
            v_counts.append(count)
            for y in starts:
                points.append((float(y + y_off), float(x + x_off)))

    all_counts = h_counts + v_counts
    mean_per_line = float(np.mean(all_counts)) if all_counts else float("nan")
    std_per_line  = float(np.std(all_counts))  if all_counts else float("nan")

    stats: Dict = {
        "num_lines":                      float(n),
        "direction":                      direction,
        "mean_intersections_per_line":    mean_per_line,
        "std_intersections_per_line":     std_per_line,
        "mean_horizontal_intersections":  float(np.mean(h_counts)) if h_counts else float("nan"),
        "mean_vertical_intersections":    float(np.mean(v_counts)) if v_counts else float("nan"),
        # Real-unit density: intersections per µm of scan line
        "mean_intersections_per_um":      None,
    }

    if um_per_px is not None and um_per_px > 0 and math.isfinite(mean_per_line):
        # Average line length in µm
        if direction in ("horizontal", "both"):
            line_length_um = W * um_per_px
        else:
            line_length_um = H * um_per_px
        if line_length_um > 0:
            stats["mean_intersections_per_um"] = mean_per_line / line_length_um

    pts = np.array(points, dtype=float) if points else np.zeros((0, 2), dtype=float)
    return stats, lines, pts


# ---------------------------------------------------------------------------
#  Transect density — from existing line geometry (editable transects)
# ---------------------------------------------------------------------------

def analyze_density_from_lines(
    mask: np.ndarray,
    horizontal_lines: List[np.ndarray],
    vertical_lines:   List[np.ndarray],
    um_per_px: Optional[float] = None,
) -> Tuple[Dict, np.ndarray]:
    """
    Compute transect density from an explicit list of line coordinates.

    Used when the user has edited (moved / deleted) transects manually.
    Lines are read from the napari Shapes layer directly — no positions
    are regenerated.

    Args:
        mask:             Label mask (binarised internally as > 0).
        horizontal_lines: List of (2, 2) arrays [[y, x0], [y, x1]].
        vertical_lines:   List of (2, 2) arrays [[y0, x], [y1, x]].
        um_per_px:        Pixel → micron scale factor.

    Returns:
        stats: Dict with mean/std intersections per line and per direction.
        pts:   (N, 2) float array of intersection entry points (y, x).
    """
    if mask.size == 0:
        return _empty_stats(), np.zeros((0, 2), dtype=float)

    mask_bin = (mask > 0).astype(np.uint8)
    H, W     = mask_bin.shape[:2]

    points:   List[Tuple[float, float]] = []
    h_counts: List[int] = []
    v_counts: List[int] = []

    def _analyze_line(coords: np.ndarray) -> Tuple[int, List[Tuple[float, float]]]:
        """Rasterize and analyze exactly the user-visible line segment."""
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return 0, []
        y0 = int(np.clip(round(float(arr[0, 0])), 0, H - 1))
        x0 = int(np.clip(round(float(arr[0, 1])), 0, W - 1))
        y1 = int(np.clip(round(float(arr[-1, 0])), 0, H - 1))
        x1 = int(np.clip(round(float(arr[-1, 1])), 0, W - 1))
        rr, cc = raster_line(y0, x0, y1, x1)
        count, starts = _count_entries(mask_bin[rr, cc])
        hits = [(float(rr[i]), float(cc[i])) for i in starts]
        return count, hits

    for coords in horizontal_lines:
        count, hits = _analyze_line(coords)
        h_counts.append(count)
        points.extend(hits)

    for coords in vertical_lines:
        count, hits = _analyze_line(coords)
        v_counts.append(count)
        points.extend(hits)

    all_counts = h_counts + v_counts
    has_h, has_v = bool(h_counts), bool(v_counts)
    direction = (
        "both"       if (has_h and has_v) else
        "horizontal" if has_h else
        "vertical"   if has_v else "none"
    )

    mean_per_line = float(np.mean(all_counts)) if all_counts else float("nan")
    std_per_line  = float(np.std(all_counts))  if all_counts else float("nan")

    stats: Dict = {
        "num_lines":                      float(len(all_counts)),
        "direction":                      direction,
        "mean_intersections_per_line":    mean_per_line,
        "std_intersections_per_line":     std_per_line,
        "mean_horizontal_intersections":  float(np.mean(h_counts)) if h_counts else float("nan"),
        "mean_vertical_intersections":    float(np.mean(v_counts)) if v_counts else float("nan"),
        "mean_intersections_per_um":      None,
    }

    if um_per_px is not None and um_per_px > 0 and math.isfinite(mean_per_line):
        # Estimate a representative line length from the mask dimensions
        if has_h:
            line_length_um = W * um_per_px
        else:
            line_length_um = H * um_per_px
        if line_length_um > 0:
            stats["mean_intersections_per_um"] = mean_per_line / line_length_um

    pts = np.array(points, dtype=float) if points else np.zeros((0, 2), dtype=float)
    return stats, pts


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _empty_stats() -> Dict:
    return {
        "num_lines":                      0.0,
        "direction":                      "none",
        "mean_intersections_per_line":    float("nan"),
        "std_intersections_per_line":     float("nan"),
        "mean_horizontal_intersections":  float("nan"),
        "mean_vertical_intersections":    float("nan"),
        "mean_intersections_per_um":      None,
    }
