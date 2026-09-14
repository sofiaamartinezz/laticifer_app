import numpy as np
import pytest
import torch

from model.inference import predict_image


class ConstantModel(torch.nn.Module):
    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, tensor):
        return torch.full_like(tensor, self.logit)


@pytest.mark.parametrize("shape", [(1, 1), (7, 11), (16, 16), (21, 29)])
def test_inference_preserves_small_and_irregular_shapes(shape):
    result = predict_image(
        ConstantModel(10.0),
        np.zeros(shape, dtype=np.uint8),
        patch_size=16,
        stride=8,
    )

    assert result.shape == shape
    assert result.dtype == np.uint8
    assert np.all(result == 255)


def test_inference_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D image"):
        predict_image(ConstantModel(0.0), np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize("patch_size,stride", [(0, 1), (16, 0), (-1, 2)])
def test_inference_rejects_invalid_window_parameters(patch_size, stride):
    with pytest.raises(ValueError, match="positive"):
        predict_image(
            ConstantModel(0.0),
            np.zeros((8, 8), dtype=np.uint8),
            patch_size=patch_size,
            stride=stride,
        )
