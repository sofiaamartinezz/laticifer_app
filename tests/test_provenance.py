from datetime import datetime

from data.provenance import analysis_timestamp, measurement_units


def test_analysis_timestamp_is_timezone_aware_iso8601():
    timestamp = datetime.fromisoformat(analysis_timestamp())

    assert timestamp.tzinfo is not None


def test_measurement_units_are_explicit():
    assert measurement_units(0.5) == ("physical", "µm", "µm²")
    assert measurement_units(None) == ("pixels_only", "px", "px²")
