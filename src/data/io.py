# data/io.py
"""
File I/O for images and masks: path inference, loading, saving.
No Qt dependency — pure pathlib / numpy / skimage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skimage import io as skio


@dataclass(frozen=True)
class ScaleCalibration:
    """A scalar pixel-size calibration and its provenance."""

    um_per_px: float
    source: str
    detail: str


MIN_REFERENCE_LINE_PX = 10.0
MIN_TYPICAL_UM_PER_PX = 0.001
MAX_TYPICAL_UM_PER_PX = 100.0


def calibration_from_reference(
    pixel_length: float,
    real_length: float,
    unit: str,
) -> ScaleCalibration:
    """Build a calibration from a user-drawn reference distance."""
    pixel_length = float(pixel_length)
    real_length = float(real_length)
    if not math.isfinite(pixel_length) or pixel_length <= 0:
        raise ValueError("The reference line must have a positive pixel length.")
    if pixel_length < MIN_REFERENCE_LINE_PX:
        raise ValueError(
            f"The reference line must be at least {MIN_REFERENCE_LINE_PX:g} pixels long. "
            "Draw a longer line for a more reliable calibration."
        )
    real_length_um = _to_micrometres(real_length, unit)
    if real_length_um is None or not math.isfinite(real_length_um):
        raise ValueError("The real-world reference length or unit is invalid.")
    detail = f"real={real_length_um:.6g} µm, pixels={pixel_length:.6g}"
    return ScaleCalibration(real_length_um / pixel_length, "reference_line", detail)


def suspicious_scale_message(um_per_px: float) -> Optional[str]:
    """Explain unusually small or large scales that merit user confirmation."""
    value = float(um_per_px)
    if not math.isfinite(value) or value <= 0:
        return "The calculated pixel size is invalid."
    if value < MIN_TYPICAL_UM_PER_PX or value > MAX_TYPICAL_UM_PER_PX:
        return (
            f"The resulting scale is 1 px = {value:.6g} µm, which is unusual. "
            "Check the entered value and unit before continuing."
        )
    return None


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


def image_source_path(image_layer) -> str:
    """Return the original image path when napari exposes one."""
    path = _source_path(image_layer)
    return str(path.resolve()) if path is not None else ""


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


def _to_micrometres(value: float, unit: str) -> Optional[float]:
    normalized = str(unit).strip().lower().replace("μ", "µ")
    factor = {
        "µm": 1.0, "um": 1.0, "micrometer": 1.0, "micrometre": 1.0,
        "nm": 0.001, "mm": 1000.0, "cm": 10000.0, "m": 1_000_000.0,
    }.get(normalized)
    return value * factor if factor is not None and value > 0 else None


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
