import numpy as np
import pytest
from skimage import io as skio

import data.batch as batch
from data.provenance import APP_VERSION


def _fake_prediction(image, device=None):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[:, image.shape[1] // 2] = 1
    return mask


def test_batch_uses_explicit_shared_scale(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    skio.imsave(input_dir / "sample.tif", np.zeros((16, 16), dtype=np.uint8))
    monkeypatch.setattr(batch, "predict_laticifer_mask", _fake_prediction)

    results = list(
        batch.run_batch_processing(
            str(input_dir),
            str(output_dir),
            run_network=False,
            um_per_px=0.5,
        )
    )

    assert len(results) == 1
    row = results[0][3]
    assert float(row["um_per_px"]) == pytest.approx(0.5)
    assert row["scale_source"] == "batch_manual"
    assert row["measurement_system"] == "physical"
    assert row["length_unit"] == "µm"
    assert row["area_unit"] == "µm²"
    assert row["transect_num_lines_per_direction"] == 10
    assert row["network_analysis_enabled"] is False
    assert row["analysis_timestamp"]
    assert row["app_version"]
    assert row["app_version"] == APP_VERSION
    assert row["source_image_path"].endswith("sample.tif")
    assert row["saved_mask_path"].endswith("sample_mask.tif")
    assert row["analysis_status"] == "success"
    assert row["error_reason"] == ""
    assert (output_dir / "masks" / "sample_mask.tif").exists()


def test_batch_can_keep_results_in_pixels(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    skio.imsave(input_dir / "plain.tif", np.zeros((16, 16), dtype=np.uint8))
    monkeypatch.setattr(batch, "predict_laticifer_mask", _fake_prediction)

    row = list(
        batch.run_batch_processing(
            str(input_dir),
            str(output_dir),
            run_network=False,
        )
    )[0][3]

    assert row["um_per_px"] == ""
    assert row["scale_source"] == "pixels_only"
    assert row["measurement_system"] == "pixels_only"
    assert row["length_unit"] == "px"
    assert row["area_unit"] == "px²"


def test_batch_records_failure_reason(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    skio.imsave(input_dir / "broken.tif", np.zeros((16, 16), dtype=np.uint8))

    def fail_prediction(image, device=None):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(batch, "predict_laticifer_mask", fail_prediction)

    row = list(batch.run_batch_processing(str(input_dir), str(output_dir)))[0][3]

    assert row["analysis_status"] == "failed"
    assert row["error_reason"] == "model unavailable"
    assert row["source_image_path"].endswith("broken.tif")


def test_batch_cancellation_stops_before_next_image(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("a.tif", "b.tif"):
        skio.imsave(input_dir / name, np.zeros((16, 16), dtype=np.uint8))

    predictions = []

    def tracked_prediction(image, device=None):
        predictions.append(1)
        return _fake_prediction(image)

    monkeypatch.setattr(batch, "predict_laticifer_mask", tracked_prediction)

    rows = list(batch.run_batch_processing(
        str(input_dir),
        str(output_dir),
        run_network=False,
        should_cancel=lambda: len(predictions) == 1,
    ))

    assert len(rows) == 1
    assert len(predictions) == 1
