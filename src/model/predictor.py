# model/predictor.py
"""
Loads the U-Net model from disk and exposes predict_laticifer_mask().
Pure numpy/torch — no Qt, no napari.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import segmentation_models_pytorch as smp

from utils.preprocessing import apply_clahe
from model.inference import predict_image as _predict_patches


MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model_soft_clDice.pth"


def _load_model(device: torch.device) -> torch.nn.Module:
    """
    Load U-Net (SE-ResNeXt50 encoder) from disk.

    Handles checkpoints saved as:
    - full nn.Module
    - plain state_dict
    - dict with 'state_dict' or 'model_state_dict'
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please place the model file there."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device).eval()

    model = smp.Unet(
        encoder_name="se_resnext50_32x4d",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}. "
            "Expected a state_dict or full nn.Module."
        )

    model.load_state_dict(state_dict)
    return model.to(device).eval()


def predict_laticifer_mask(image: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """
    Predict a binary laticifer mask using patch-based U-Net inference.

    Args:
        image:  Raw image array (RGB or grayscale, any dtype).
        device: 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        Binary uint8 mask (values 0/1) ready for a napari Labels layer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}.")

    preprocessed = apply_clahe(img)   # → grayscale uint8 (H, W)
    pred_255 = _predict_patches(
        model=_load_model(torch.device(device)),
        image_np=preprocessed,
        patch_size=512,
        stride=256,
        threshold=0.5,
        device=device,
    )
    return (pred_255 > 127).astype(np.uint8)