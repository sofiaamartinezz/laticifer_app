from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from data.io import read_pixel_size_from_path, read_pixel_size_metadata


def test_reads_isotropic_ome_pixel_size(tmp_path):
    path = tmp_path / "scaled.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 8), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.65,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.65,
            "PhysicalSizeYUnit": "µm",
        },
    )

    calibration, error = read_pixel_size_from_path(path)

    assert error == ""
    assert calibration is not None
    assert calibration.um_per_px == pytest.approx(0.65)
    assert calibration.source == "ome_metadata"


def test_reads_standard_tiff_resolution(tmp_path):
    path = tmp_path / "resolution.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 8), dtype=np.uint8),
        resolution=(100, 100),
        resolutionunit="CENTIMETER",
    )

    calibration, error = read_pixel_size_from_path(path)

    assert error == ""
    assert calibration is not None
    assert calibration.um_per_px == pytest.approx(100.0)
    assert calibration.source == "tiff_resolution"


def test_rejects_anisotropic_ome_pixels(tmp_path):
    path = tmp_path / "anisotropic.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 8), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )

    calibration, error = read_pixel_size_from_path(path)

    assert calibration is None
    assert "anisotropic" in error.lower()


def test_layer_without_source_path_has_clear_error():
    layer = SimpleNamespace(source=None, metadata={})

    calibration, error = read_pixel_size_metadata(layer)

    assert calibration is None
    assert "file path" in error
