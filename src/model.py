# model.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import segmentation_models_pytorch as smp

# Make sure utils is importable if running as scripts
import sys
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.preprocessing import apply_clahe
from utils.inference import predict_image as predict_image_patches


MODEL_PATH = Path(__file__).parent / "models" / "best_model_soft_clDice.pth"


def _load_unet_model(device: torch.device) -> torch.nn.Module:
    """
    Load U-Net model with SE-ResNeXt50 encoder from disk.

    Handles checkpoints saved as:
    - full nn.Module
    - plain state_dict
    - dict with 'state_dict' or 'model_state_dict'
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please ensure your model is saved there."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model = smp.Unet(
            encoder_name="se_resnext50_32x4d",
            encoder_weights=None,
            in_channels=1,
            classes=1,
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError(
                f"Unsupported checkpoint type: {type(checkpoint)}. "
                "Expected a state_dict or full nn.Module."
            )

        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def predict_laticifer_mask(image: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """
    Predict a binary laticifer mask using a U-Net model with patch-based inference.

    Steps:
    - Take the raw image (RGB or grayscale)
    - Apply CLAHE-based preprocessing (returns grayscale uint8)
    - Run patch-based prediction (0–255 mask)
    - Return a 0/1 uint8 mask for napari Labels
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = _load_unet_model(torch_device)

    # Ensure we have a NumPy array
    img = np.asarray(image)

    # Basic sanity check
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D image, got shape {img.shape} (ndim = {img.ndim})."
        )

    # CLAHE preprocessing (handles RGB/gray and dtype internally)
    preprocessed = apply_clahe(img)  # → grayscale uint8 (H, W)

    # Patch-based prediction: returns 0–255 uint8
    pred_255 = predict_image_patches(
        model=model,
        image_np=preprocessed,
        patch_size=512,
        stride=256,
        threshold=0.5,
        device=device,
    )

    # Convert 0/255 → 0/1 labels for napari
    binary_mask = (pred_255 > 127).astype(np.uint8)

    print(
        f"[DEBUG] Mask generated: shape={binary_mask.shape}, dtype={binary_mask.dtype}, "
        f"unique_values={np.unique(binary_mask)}"
    )

    return binary_mask
