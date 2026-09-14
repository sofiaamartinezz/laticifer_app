# model/inference.py
"""
Patch-based sliding window inference for the U-Net model.
Pure numpy/torch — no Qt, no napari.
"""
import numpy as np
import torch
import torch.nn.functional as F


def predict_image(
    model,
    image_np: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    threshold: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run sliding-window inference over a full image, padding with reflection
    so every pixel is covered, then crop back to the original size.

    Returns a uint8 mask with values 0 or 255.
    """
    model.to(device)
    model.eval()

    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    if image_tensor.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {tuple(image_tensor.shape)}.")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers.")
    H, W = image_tensor.shape
    if H == 0 or W == 0:
        raise ValueError("Cannot run inference on an empty image.")

    def _padded_size(length: int) -> int:
        if length <= patch_size:
            return patch_size
        return ((length - patch_size + stride - 1) // stride) * stride + patch_size

    H_pad, W_pad = _padded_size(H), _padded_size(W)

    pad_h, pad_w = H_pad - H, W_pad - W
    # Reflection padding requires every padding amount to be smaller than the
    # corresponding input dimension. Tiny crops therefore need replication.
    pad_mode = "reflect" if pad_h < H and pad_w < W else "replicate"
    image_tensor = F.pad(
        image_tensor.unsqueeze(0).unsqueeze(0) / 255.0,
        (0, pad_w, 0, pad_h),
        mode=pad_mode,
    ).to(device)  # (1, 1, H_pad, W_pad)

    output    = torch.zeros((1, 1, H_pad, W_pad), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, 1, H_pad, W_pad), dtype=torch.float32, device=device)

    for top in range(0, H_pad - patch_size + 1, stride):
        for left in range(0, W_pad - patch_size + 1, stride):
            patch = image_tensor[:, :, top:top + patch_size, left:left + patch_size]
            with torch.no_grad():
                pred = model(patch)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                pred = torch.sigmoid(pred)
            output   [:, :, top:top + patch_size, left:left + patch_size] += pred
            count_map[:, :, top:top + patch_size, left:left + patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    output = output / count_map

    binary = (output[:, :, :H, :W] > threshold).float()
    return (binary[0, 0].cpu().numpy() * 255).astype(np.uint8)
