# utils/scalebar.py
"""
Scale bar auto-detection from microscopy images.

The scale bar must be *baked into the image pixels* — i.e. rendered by the
acquisition software before saving.  It will NOT be found if it is only a
napari viewer overlay.

Public API
──────────
detect_scalebar(image, hints)  →  (ScalebarResult | None, message: str)

ScalebarResult carries both the pixel width of the bar and its full bounding
box in image coordinates so the caller can draw a highlighting layer in napari.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

@dataclass
class ScalebarHints:
    """
    User-supplied hints that guide the detector.  All fields have sensible
    defaults so the caller can pass ScalebarHints() with no arguments and
    detection will still run.

    Attributes
    ----------
    color : str
        'white'  – bright bar on a dark background (fluorescence microscopy).
        'black'  – dark bar on a bright background (brightfield / transmitted).
        'auto'   – try white first, then black; use when unsure.
    region : str
        Spatial region to search.  Limiting the region avoids false positives
        from tissue structure.
        'bottom' | 'top' | 'left' | 'right' | 'any'
    search_fraction : float
        What fraction of the image to include in the search region.
        0.20 = bottom (or top / left / right) 20 % of the image.
        Increase this if the bar is further from the edge.
    min_width_px : int
        Minimum horizontal extent in pixels.  Candidates shorter than this
        are discarded.  Lower for small / low-resolution images.
    min_aspect_ratio : float
        Minimum width-to-height ratio.  A real scale bar is always much wider
        than it is tall; values ≥ 5 are appropriate.
    brightness_percentile : float
        Aggressiveness of thresholding.
        White bars: pixels above this percentile are candidates.
        Black bars:  pixels below (100 − percentile) are candidates.
        Lower values = less selective (catches dimmer bars but more noise).
    """
    color:                 str   = "auto"
    region:                str   = "bottom"
    search_fraction:       float = 0.20
    min_width_px:          int   = 30
    min_aspect_ratio:      float = 5.0
    brightness_percentile: float = 90.0


@dataclass
class ScalebarResult:
    """
    Everything the caller needs after a successful detection.

    Attributes
    ----------
    width_px  : Horizontal length of the scale bar in pixels.
    y0, y1    : Row range of the bar in the *full* image (for layer display).
    x0, x1    : Column range of the bar in the *full* image.
    color     : Which color polarity succeeded ('white' or 'black').
    """
    width_px: int
    y0: int
    y1: int
    x0: int
    x1: int
    color: str


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert any image array to a float32 2-D grayscale.

    RGB/RGBA channels are averaged so both tissue signal and a neutral
    (white or black) scale bar are captured equally.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]          # drop alpha
        return img.mean(axis=-1)        # average R, G, B
    return img                          # already single-channel


def _crop_region(
    gray: np.ndarray,
    region: str,
    fraction: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Crop `gray` to the requested spatial region.

    Returns
    -------
    crop  : 2-D slice of `gray`
    y_off : row offset of crop[0, 0] within the full image
    x_off : col offset of crop[0, 0] within the full image
    """
    H, W = gray.shape
    f = float(np.clip(fraction, 0.01, 1.0))

    if region == "bottom":
        y0 = max(0, int(H * (1.0 - f)))
        return gray[y0:, :],    y0, 0
    if region == "top":
        y1 = min(H, int(H * f))
        return gray[:y1, :],     0, 0
    if region == "right":
        x0 = max(0, int(W * (1.0 - f)))
        return gray[:, x0:],     0, x0
    if region == "left":
        x1 = min(W, int(W * f))
        return gray[:, :x1],     0, 0
    # 'any' – full image
    return gray, 0, 0


def _threshold_and_label(
    crop: np.ndarray,
    color: str,
    pct: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Threshold the crop for one color polarity and label connected components.

    Returns
    -------
    labeled   : integer label array (or None on failure)
    binary    : boolean mask that was labelled (or None on failure)
    reason    : human-readable description of what happened
    """
    if color == "white":
        # Pixels that are clearly brighter than most of the search region
        threshold = float(np.percentile(crop, pct))
        median    = float(np.median(crop))
        # Guard: if the region is already uniformly bright, thresholding
        # won't isolate anything meaningful
        if threshold - median < 5:
            return None, None, (
                f"Search region is too uniform for a white bar "
                f"(max={crop.max():.0f}, threshold={threshold:.0f}, "
                f"median={median:.0f}). "
                "Try a different region or switch to black bar."
            )
        binary = crop >= threshold

    else:  # 'black'
        threshold = float(np.percentile(crop, 100.0 - pct))
        # Need meaningful contrast so we can distinguish bar from background
        if crop.max() - crop.min() < 10:
            return None, None, (
                "Search region has insufficient contrast for a black bar "
                f"(range = {crop.max() - crop.min():.0f} grey levels)."
            )
        # Only keep pixels that are dark AND have a brighter neighbourhood.
        # This avoids confusing the bar with a globally dark tissue region
        # (common in fluorescence images where the background is near-black).
        local_max = ndi.maximum_filter(crop, size=15)
        dark      = crop <= threshold
        near_bright = local_max > (threshold + 10)
        binary    = dark & near_bright

    labeled, n = ndi.label(binary)
    if n == 0:
        return None, None, (
            f"No {color} regions found after thresholding "
            f"(threshold ≈ {threshold:.0f})."
        )
    return labeled, binary, f"threshold={threshold:.0f}, {n} component(s) found"


def _best_component(
    labeled: np.ndarray,
    y_off: int,
    x_off: int,
    min_width_px: int,
    min_aspect: float,
) -> Tuple[Optional[ScalebarResult], str, int]:
    """
    Walk through labeled components and pick the most elongated one that
    meets the minimum size and aspect-ratio criteria.

    Returns
    -------
    result       : ScalebarResult if found, else None
    reason       : human-readable explanation
    n_candidates : number of components that passed the geometry filter
    """
    n_labels = int(labeled.max())
    best: Optional[ScalebarResult] = None
    best_aspect = 0.0
    n_candidates = 0

    for lbl in range(1, n_labels + 1):
        ys, xs = np.where(labeled == lbl)
        width  = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)

        if width < min_width_px:
            continue                       # too short
        if height == 0:
            continue
        aspect = width / height
        if aspect < min_aspect:
            continue                       # not flat enough to be a bar

        n_candidates += 1
        if best is None or aspect > best_aspect:
            best_aspect = aspect
            best = ScalebarResult(
                width_px = width,
                y0       = int(ys.min()) + y_off,
                y1       = int(ys.max()) + y_off,
                x0       = int(xs.min()) + x_off,
                x1       = int(xs.max()) + x_off,
                color    = "",             # filled in by caller
            )

    if best is None:
        return None, (
            f"{n_labels} component(s) found but none had "
            f"width ≥ {min_width_px} px and aspect ratio ≥ {min_aspect:.0f}. "
            "Try: lower the minimum width, widen the search region, "
            "or change the bar color."
        ), 0

    return best, f"best aspect ratio {best_aspect:.1f}", n_candidates


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def detect_scalebar(
    image: np.ndarray,
    hints: Optional[ScalebarHints] = None,
) -> Tuple[Optional[ScalebarResult], str]:
    """
    Detect a scale bar baked into a microscopy image.

    The bar must be rendered into the pixel data by the acquisition software.
    Napari viewer overlays are not part of the pixel data and will not be
    detected.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale or 3-D RGB/RGBA image (uint8 or float).
    hints : ScalebarHints, optional
        User-supplied search parameters.  Pass None to use defaults.

    Returns
    -------
    (result, message)
        result  : ScalebarResult if a bar was found, else None.
        message : Human-readable outcome — always populated, shown in the UI
                  so the user knows what to do if detection fails.
    """
    if hints is None:
        hints = ScalebarHints()

    gray               = _to_grayscale(image)
    crop, y_off, x_off = _crop_region(gray, hints.region, hints.search_fraction)

    # Which polarities to try
    colors: List[str] = ["white", "black"] if hints.color == "auto" else [hints.color]

    last_message = "No detection attempted."

    for color in colors:
        labeled, binary, thresh_msg = _threshold_and_label(
            crop, color, hints.brightness_percentile
        )
        if labeled is None:
            last_message = f"[{color}] {thresh_msg}"
            continue

        result, geom_msg, n_cand = _best_component(
            labeled, y_off, x_off,
            hints.min_width_px, hints.min_aspect_ratio,
        )
        if result is None:
            last_message = f"[{color}] {thresh_msg} → {geom_msg}"
            continue

        result.color = color
        return result, (
            f"Detected {color} scale bar: {result.width_px} px wide "
            f"(x: {result.x0}–{result.x1}, y: {result.y0}–{result.y1}). "
            f"{n_cand} candidate(s); {geom_msg}."
        )

    return None, last_message