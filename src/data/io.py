# data/io.py
"""
File I/O for images and masks: path inference, loading, saving.
No Qt dependency — pure pathlib / numpy / skimage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skimage import io as skio


# -----------------------------------------------------------------------------
#  Path inference
# -----------------------------------------------------------------------------

def infer_dataset_root(image_layer) -> Optional[Path]:
    """
    Try to auto-detect the dataset root from the image layer's source path.
    Walks up to 5 parent directories looking for a folder named 'dataset'.
    """
    img_path = _source_path(image_layer)
    if img_path is None:
        return None
    p = img_path
    for _ in range(5):
        if p.name == "dataset":
            return p
        p = p.parent
    return None


def infer_mask_path(image_layer, dataset_root: Optional[Path]) -> Optional[Path]:
    """
    Try to guess the mask path for a given image layer.

    Checks in order:
    1. dataset_root/masks/<stem>_mask.tif
    2. Sibling masks/ folder when image lives under .../images/
    """
    if image_layer is None:
        return None

    stem = Path(image_layer.name).stem if image_layer.name else None
    if stem and dataset_root is not None:
        candidate = dataset_root / "masks" / f"{stem}_mask.tif"
        if candidate.exists():
            return candidate

    img_path = _source_path(image_layer)
    if img_path is not None and img_path.parent.name == "images":
        candidate = img_path.parent.parent / "masks" / f"{img_path.stem}_mask.tif"
        if candidate.exists():
            return candidate

    return None


def _source_path(layer) -> Optional[Path]:
    """Extract a filesystem path from a napari layer, if available."""
    src = getattr(layer, "source", None)
    if src is not None and getattr(src, "path", None):
        try:
            return Path(src.path)
        except TypeError:
            pass
    if isinstance(getattr(layer, "metadata", None), dict):
        for key in ("source", "filename", "file_name", "file_path"):
            if key in layer.metadata:
                try:
                    return Path(layer.metadata[key])
                except TypeError:
                    continue
    return None


# -----------------------------------------------------------------------------
#  Loading
# -----------------------------------------------------------------------------

def load_mask(mask_path: Path, expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Load a mask from disk, ensure 2D, validate shape, return 0/1 uint8.
    Returns None on any error.
    """
    try:
        mask_img = skio.imread(mask_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load mask: {exc}")
        return None

    if mask_img.ndim == 3:
        mask_img = mask_img[..., 0]

    if mask_img.shape != tuple(expected_shape):
        print(f"[WARN] Shape mismatch: mask {mask_img.shape} vs image {expected_shape}")
        return None

    return (mask_img > 0).astype(np.uint8)


# -----------------------------------------------------------------------------
#  Saving
# -----------------------------------------------------------------------------

def save_image_and_mask(
    image_data: np.ndarray,
    mask_labels: np.ndarray,
    base_name: str,
    dataset_root: Path,
) -> Tuple[Path, Path]:
    """
    Save image and binary mask (stored as 0/255) under dataset_root.

    Creates dataset_root/images/ and dataset_root/masks/ if needed.
    Returns (image_out_path, mask_out_path).
    """
    images_dir = dataset_root / "images"
    masks_dir  = dataset_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_out = images_dir / f"{base_name}.tif"
    mask_out  = masks_dir  / f"{base_name}_mask.tif"

    skio.imsave(image_out, image_data)
    skio.imsave(mask_out, (mask_labels > 0).astype(np.uint8) * 255)

    return image_out, mask_out