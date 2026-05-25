# utils/postprocessing.py
"""
Post-processing operations applied to binary laticifer masks.
All functions accept any mask dtype (values > 0 are foreground) and
return an array of the same dtype.

Pure numpy / skimage — no Qt, no napari.
"""
from __future__ import annotations

import numpy as np
from skimage import morphology


def fill_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """
    Fill holes inside foreground objects that are smaller than area_threshold pixels.

    Args:
        mask:            Label mask (values > 0 are laticifer).
        area_threshold:  Maximum hole size to fill (pixels²).

    Returns:
        Mask of the same dtype with small holes closed.
    """
    mask_bool = mask > 0
    cleaned   = morphology.remove_small_holes(mask_bool, area_threshold=area_threshold)
    return cleaned.astype(mask.dtype)


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove disconnected foreground objects smaller than min_size pixels.

    Args:
        mask:     Label mask (values > 0 are laticifer).
        min_size: Minimum object size to keep (pixels).

    Returns:
        Mask of the same dtype with small objects removed.
    """
    mask_bool = mask > 0
    cleaned   = morphology.remove_small_objects(mask_bool, min_size=min_size)
    return cleaned.astype(mask.dtype)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reduce the mask to a 1-pixel-wide skeleton (centreline).
    Useful for visualising network topology or as input to network analysis.

    Args:
        mask: Label mask (values > 0 are laticifer).

    Returns:
        Boolean skeleton cast to the original mask dtype.
    """
    mask_bool = mask > 0
    skel      = morphology.skeletonize(mask_bool)
    return skel.astype(mask.dtype)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Expand foreground regions by `radius` pixels (morphological dilation).

    Args:
        mask:   Label mask (values > 0 are laticifer).
        radius: Structuring element radius in pixels.

    Returns:
        Dilated mask of the same dtype.
    """
    mask_bool = mask > 0
    selem     = morphology.disk(radius)
    cleaned   = morphology.binary_dilation(mask_bool, footprint=selem)
    return cleaned.astype(mask.dtype)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Shrink foreground regions by `radius` pixels (morphological erosion).

    Args:
        mask:   Label mask (values > 0 are laticifer).
        radius: Structuring element radius in pixels.

    Returns:
        Eroded mask of the same dtype.
    """
    mask_bool = mask > 0
    selem     = morphology.disk(radius)
    cleaned   = morphology.binary_erosion(mask_bool, footprint=selem)
    return cleaned.astype(mask.dtype)