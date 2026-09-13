import pytest

from data.io import calibration_from_reference, suspicious_scale_message


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


@pytest.mark.parametrize("pixel_length", [0.1, 5, 9.99])
def test_reference_calibration_rejects_short_line(pixel_length):
    with pytest.raises(ValueError, match="at least 10 pixels"):
        calibration_from_reference(pixel_length, 100, "µm")


@pytest.mark.parametrize("um_per_px", [0.0001, 101.0])
def test_extreme_scale_requires_warning(um_per_px):
    assert suspicious_scale_message(um_per_px) is not None


@pytest.mark.parametrize("um_per_px", [0.001, 0.5, 100.0])
def test_typical_scale_does_not_require_warning(um_per_px):
    assert suspicious_scale_message(um_per_px) is None
