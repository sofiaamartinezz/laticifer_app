# utils/preprocessing.py
import numpy as np


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to an image and return an enhanced grayscale uint8 image.

    - Accepts:
        * 2D grayscale images (H, W)
        * 3D RGB images (H, W, 3 or 4)
    - If RGB, it converts to grayscale by **extracting the Green channel**.
    - If float, it is scaled to 0–255 and cast to uint8.

    Note:
    cv2 is imported lazily inside the function to avoid Qt plugin
    conflicts when using napari.
    """
    import cv2  # lazy import so it doesn't crush napari Qt plugins

    img = np.asarray(image)

    # If RGB/RGBA, extract Green Channel
    if img.ndim == 3:
        # If it has an alpha channel, drop it
        if img.shape[-1] == 4:
            img = img[..., :3]

        # Extract Green channel (Index 1)
        # Note: Whether RGB (skimage/napari) or BGR (opencv), Green is always index 1.
        if img.shape[-1] >= 2:
            img_gray = img[..., 1]
        else:
            # Fallback if shape is (H, W, 1)
            img_gray = img[..., 0]
    else:
        # Already single-channel 2D
        img_gray = img

    # Ensure uint8 range 0–255
    if img_gray.dtype in (np.float32, np.float64):
        g = img_gray.astype(np.float32)
        # If values are small (0-1), scale them up
        if g.max() <= 1.0:
            g = g * 255.0
        # Clip to safe bounds and cast
        img_gray = np.clip(g, 0, 255).astype(np.uint8)
    elif img_gray.dtype != np.uint8:
        # Simple clip + cast for other integer types (e.g. uint16)
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    # Apply CLAHE on grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    return enhanced