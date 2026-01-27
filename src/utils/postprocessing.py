# src/utils/postprocessing.py
import numpy as np
from skimage import morphology

def fill_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """
    Fills holes within the foreground objects that are smaller than the threshold.
    """
    # Convert to boolean for morphology
    mask_bool = mask > 0
    # remove_small_holes fills holes in the *True* regions
    cleaned = morphology.remove_small_holes(mask_bool, area_threshold=area_threshold)
    return cleaned.astype(mask.dtype)

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reduces the mask to a 1-pixel wide skeleton.
    Useful for visualizing the topology of the network.
    """
    mask_bool = mask > 0
    # skeletonize returns a boolean array
    skel = morphology.skeletonize(mask_bool)
    return skel.astype(mask.dtype)

def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Removes disconnected foreground objects smaller than min_size pixels.
    """
    mask_bool = mask > 0
    cleaned = morphology.remove_small_objects(mask_bool, min_size=min_size)
    return cleaned.astype(mask.dtype)

def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Expands the white regions by 'radius' pixels.
    """
    mask_bool = mask > 0
    selem = morphology.disk(radius)
    cleaned = morphology.binary_dilation(mask_bool, footprint=selem)
    return cleaned.astype(mask.dtype)

def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Shrinks the white regions by 'radius' pixels.
    """
    mask_bool = mask > 0
    selem = morphology.disk(radius)
    cleaned = morphology.binary_erosion(mask_bool, footprint=selem)
    return cleaned.astype(mask.dtype)