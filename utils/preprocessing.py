# utils/preprocessing.py
import numpy as np


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to an image and return an enhanced grayscale uint8 image.

    - Accepts:
        * 2D grayscale images (H, W)
        * 3D RGB images (H, W, 3 or 4)
    - If RGB, it is converted to grayscale.
    - If float, it is scaled to 0–255 and cast to uint8.

    Note:
    cv2 is imported lazily inside the function to avoid Qt plugin
    conflicts when using napari.
    """
    import cv2  # lazy import so it doesn't crush napari Qt plugins

    img = np.asarray(image)

    # If RGB/RGBA, convert to grayscale
    if img.ndim == 3:
        # If it has an alpha channel, drop it
        if img.shape[-1] == 4:
            img = img[..., :3]

        # Heuristic: assume RGB order (as used by skimage/napari)
        # If you ever feed BGR images from cv2.imread, convert before calling.
        if img.dtype in (np.float32, np.float64):
            # assume 0–1 or 0–255, normalize to 0–1 for cv2
            img_float = img.astype(np.float32)
            if img_float.max() > 1.0:
                img_float = img_float / 255.0
            img_rgb_01 = img_float
            img_gray = cv2.cvtColor((img_rgb_01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img_rgb = img.astype(np.uint8)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        # Already single-channel
        img_gray = img

    # Ensure uint8 range 0–255
    if img_gray.dtype in (np.float32, np.float64):
        g = img_gray.astype(np.float32)
        if g.max() <= 1.0:
            g = g * 255.0
        img_gray = g.astype(np.uint8)
    elif img_gray.dtype != np.uint8:
        # Simple clip + cast
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    # Apply CLAHE on grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    return enhanced
