# utils/preprocessing.py
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
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

    # Convert to uint8 without clipping high-bit-depth microscopy images.
    # Using the observed finite range also handles 10/12-bit data stored in a
    # uint16 container, where dividing by 65535 would waste most of the range.
    if img_gray.dtype != np.uint8:
        g = img_gray.astype(np.float32)
        finite = np.isfinite(g)
        if not np.any(finite):
            img_gray = np.zeros(g.shape, dtype=np.uint8)
        else:
            lo = float(np.min(g[finite]))
            hi = float(np.max(g[finite]))
            if hi <= lo:
                fill = np.clip(lo, 0, 255)
                img_gray = np.full(g.shape, fill, dtype=np.uint8)
            else:
                scaled = (g - lo) * (255.0 / (hi - lo))
                scaled[~finite] = 0.0
                img_gray = np.clip(scaled, 0, 255).astype(np.uint8)

    # Apply CLAHE on grayscale
    if clip_limit <= 0 or tile_grid_size <= 0:
        raise ValueError("CLAHE clip limit and tile size must be positive.")
    tile_size = int(tile_grid_size)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(tile_size, tile_size)
    )
    enhanced = clahe.apply(img_gray)

    return enhanced
