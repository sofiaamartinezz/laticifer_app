# utils/tiled_inference.py
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
    Patch-based sliding window inference with padding so that
    the whole image is covered, then cropped back to (H, W).

    Returns uint8 mask 0/255.
    """
    model.to(device)
    model.eval()

    # Ensure float32 tensor
    if image_np.dtype != np.float32:
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
    else:
        image_tensor = torch.from_numpy(image_np)

    H, W = image_tensor.shape

    # Compute padded size so sliding window fully covers the image
    if H <= patch_size:
        H_pad = patch_size
    else:
        H_pad = ((H - patch_size + stride - 1) // stride) * stride + patch_size

    if W <= patch_size:
        W_pad = patch_size
    else:
        W_pad = ((W - patch_size + stride - 1) // stride) * stride + patch_size

    pad_bottom = H_pad - H
    pad_right = W_pad - W

    # Pad (left, right, top, bottom) = (0, pad_right, 0, pad_bottom)
    image_tensor = F.pad(
        image_tensor.unsqueeze(0).unsqueeze(0) / 255.0,
        (0, pad_right, 0, pad_bottom),
        mode="reflect",
    )  # (1, 1, H_pad, W_pad)

    image_tensor = image_tensor.to(device)

    _, _, H_pad, W_pad = image_tensor.shape

    output = torch.zeros((1, 1, H_pad, W_pad), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, 1, H_pad, W_pad), dtype=torch.float32, device=device)

    # Sliding window over padded image
    for top in range(0, H_pad - patch_size + 1, stride):
        for left in range(0, W_pad - patch_size + 1, stride):
            patch = image_tensor[:, :, top:top + patch_size, left:left + patch_size]

            with torch.no_grad():
                pred = model(patch)

                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                if not torch.is_tensor(pred):
                    raise TypeError(
                        f"Model output must be a Tensor, got {type(pred)} instead."
                    )

                pred = torch.sigmoid(pred)

            output[:, :, top:top + patch_size, left:left + patch_size] += pred
            count_map[:, :, top:top + patch_size, left:left + patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    output /= count_map

    # Crop back to original size
    output = output[:, :, :H, :W]

    binary_mask = (output > threshold).float()
    mask_np = binary_mask.squeeze().cpu().numpy() * 255.0

    return mask_np.astype(np.uint8)
