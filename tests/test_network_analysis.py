import math

import numpy as np
import pytest

from utils.network_analysis import run_network_analysis


def test_horizontal_skeleton_uses_edge_length():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 1:6] = 1

    stats, _ = run_network_analysis(mask, um_per_px=2.0)

    assert stats.total_skeleton_length_px == pytest.approx(4.0)
    assert stats.total_skeleton_length_um == pytest.approx(8.0)
    assert stats.endpoint_count == 2


def test_diagonal_skeleton_uses_euclidean_steps():
    mask = np.eye(5, dtype=np.uint8)

    stats, _ = run_network_analysis(mask)

    assert stats.total_skeleton_length_px == pytest.approx(4 * math.sqrt(2))
    assert stats.endpoint_count == 2


def test_empty_mask_returns_zero_network():
    stats, geometry = run_network_analysis(np.zeros((8, 8), dtype=np.uint8))

    assert stats.total_skeleton_length_px == 0.0
    assert stats.connected_components == 0
    assert geometry.bifurcation_points.shape == (0, 2)
