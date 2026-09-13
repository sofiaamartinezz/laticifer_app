import numpy as np

from utils.quantification import (
    analyze_density_from_lines,
    analyze_density_pixel_ratio,
    analyze_density_transect,
)


def test_pixel_ratio_and_physical_areas():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:2, :] = 1

    stats = analyze_density_pixel_ratio(mask, um_per_px=0.5)

    assert stats["laticifer_pixels"] == 20
    assert stats["density_percentage"] == 20.0
    assert stats["laticifer_area_um2"] == 5.0
    assert stats["total_area_um2"] == 25.0


def test_transect_counts_foreground_run_at_image_boundary():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, :2] = 1

    stats, _, points = analyze_density_transect(
        mask, num_lines=1, direction="horizontal"
    )

    assert stats["mean_intersections_per_line"] == 1.0
    assert points.tolist() == [[2.0, 0.0]]


def test_edited_transect_uses_only_its_visible_segment():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 5] = 1
    short_line = [np.array([[3.0, 0.0], [3.0, 2.0]])]

    stats, points = analyze_density_from_lines(mask, short_line, [])

    assert stats["mean_intersections_per_line"] == 0.0
    assert points.shape == (0, 2)


def test_diagonal_transect_is_rasterized_between_endpoints():
    mask = np.eye(7, dtype=np.uint8)
    diagonal = [np.array([[0.0, 0.0], [6.0, 6.0]])]

    stats, points = analyze_density_from_lines(mask, diagonal, [])

    assert stats["mean_intersections_per_line"] == 1.0
    assert points.tolist() == [[0.0, 0.0]]
