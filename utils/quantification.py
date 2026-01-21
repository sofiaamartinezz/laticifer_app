# utils/quantification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# pixel ratio
# ----------------------------

def analyze_density_pixel_ratio(mask: np.ndarray, roi: Optional[Tuple[slice, slice]] = None) -> Dict[str, float]:
    """
    Density by area fraction: laticifer_pixels / total_pixels.
    mask: labels or binary; any value > 0 counts as mask.
    roi: optional (yslice, xslice) to restrict analysis.
    """
    if mask.size == 0:
        return {
            "laticifer_pixels": 0,
            "total_pixels": 0,
            "pixel_ratio": float("nan"),
            "density_percentage": float("nan"),
        }

    data = mask
    if roi is not None:
        ysl, xsl = roi
        data = data[ysl, xsl]

    total_pixels = int(data.size)
    if total_pixels == 0:
        return {
            "laticifer_pixels": 0,
            "total_pixels": 0,
            "pixel_ratio": float("nan"),
            "density_percentage": float("nan"),
        }

    laticifer_pixels = int(np.count_nonzero(data > 0))
    ratio = laticifer_pixels / total_pixels
    return {
        "laticifer_pixels": laticifer_pixels,
        "total_pixels": total_pixels,
        "pixel_ratio": float(ratio),
        "density_percentage": float(ratio * 100.0),
    }


# ==========================================================
# transect function 
# ==========================================================

def analyze_density_transect(
    mask: np.ndarray,
    num_lines: int = 10,
    direction: str = "both",
    roi: Optional[Tuple[slice, slice]] = None,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
) -> Tuple[Dict[str, float], List[np.ndarray], np.ndarray]:
    """
    Replicates exactly the logic from your standalone script:

    - mask is binarized as (mask > 0)
    - line positions are np.linspace(start_frac*H, end_frac*H, num_lines) (and similarly for W)
    - for each line: diffs = np.diff(line_pixels.astype(np.int8))
      intersections counted as count_nonzero(diffs == 1)  # entries 0->1

    Returns:
      stats: dict with same style metrics as your script
      lines: list of (2,2) arrays for napari Shapes lines (y,x)
      pts: (N,2) array for napari Points (y,x) (entry points 0->1)
    """
    if mask.size == 0:
        return {"error": "Empty mask"}, [], np.zeros((0, 2), dtype=float)

    # binarize exactly as script (but with napari labels: >0)
    mask_bin = (mask > 0).astype(np.uint8)

    # Apply ROI crop; keep offset so we draw overlays in global coords
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

    # --- HORIZONTAL ---
    if direction in ("horizontal", "both"):
        y_positions = np.linspace(start_frac * H, end_frac * H, n, dtype=int)
        for y in y_positions:
            y = int(np.clip(y, 0, H - 1))

            # napari line geometry in global coords (y,x)
            y_g = float(y + y_off)
            lines.append(np.array([[y_g, float(0 + x_off)], [y_g, float((W - 1) + x_off)]], dtype=float))

            line_pixels = mask_bin[y, :]
            diffs = np.diff(line_pixels.astype(np.int8))

            num_intersections = int(np.count_nonzero(diffs == 1))  # EXACTLY like script
            horizontal_intersections.append(num_intersections)

            xs = np.where(diffs == 1)[0] + 1
            for x in xs:
                points.append((float(y + y_off), float(x + x_off)))  # (y,x)

    # --- VERTICAL ---
    if direction in ("vertical", "both"):
        x_positions = np.linspace(start_frac * W, end_frac * W, n, dtype=int)
        for x in x_positions:
            x = int(np.clip(x, 0, W - 1))

            x_g = float(x + x_off)
            lines.append(np.array([[float(0 + y_off), x_g], [float((H - 1) + y_off), x_g]], dtype=float))

            line_pixels = mask_bin[:, x]
            diffs = np.diff(line_pixels.astype(np.int8))

            num_intersections = int(np.count_nonzero(diffs == 1))  # EXACTLY like script
            vertical_intersections.append(num_intersections)

            ys = np.where(diffs == 1)[0] + 1
            for y in ys:
                points.append((float(y + y_off), float(x + x_off)))  # (y,x)

    all_intersections = horizontal_intersections + vertical_intersections

    stats: Dict[str, float] = {
        "num_lines": float(n),
        "direction": direction,  # string; UI can display directly
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