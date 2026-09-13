import pytest

from data.io import calibration_from_reference


@pytest.mark.parametrize(
    "real_length,unit,expected_um_per_px",
    [(100, "µm", 0.5), (100_000, "nm", 0.5), (0.1, "mm", 0.5)],
)
def test_reference_calibration_converts_units(
    real_length, unit, expected_um_per_px
):
    calibration = calibration_from_reference(200, real_length, unit)

    assert calibration.um_per_px == pytest.approx(expected_um_per_px)
    assert calibration.source == "reference_line"


@pytest.mark.parametrize("pixel_length", [0, -1, float("nan")])
def test_reference_calibration_rejects_invalid_line(pixel_length):
    with pytest.raises(ValueError, match="positive pixel length"):
        calibration_from_reference(pixel_length, 100, "µm")
