import numpy as np
import pytest

from utils.preprocessing import apply_clahe


def test_uint16_image_is_scaled_without_saturating_everything():
    image = np.linspace(0, 4095, 64 * 64, dtype=np.uint16).reshape(64, 64)

    result = apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.unique(result).size > 16


def test_rgb_uses_green_channel_and_returns_grayscale():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 1] = np.arange(64, dtype=np.uint8)[None, :] * 4

    result = apply_clahe(image)

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8


def test_non_finite_float_image_is_handled():
    image = np.full((32, 32), np.nan, dtype=np.float32)

    result = apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("clip_limit,tile_size", [(0, 8), (2, 0)])
def test_clahe_rejects_invalid_settings(clip_limit, tile_size):
    with pytest.raises(ValueError, match="must be positive"):
        apply_clahe(
            np.zeros((8, 8), dtype=np.uint8),
            clip_limit=clip_limit,
            tile_grid_size=tile_size,
        )
