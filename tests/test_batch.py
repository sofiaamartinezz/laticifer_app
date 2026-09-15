import numpy as np
import pytest
from pathlib import Path
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
    assert row["saved_mask_path"].endswith("sample_tif_mask.tif")
    assert row["analysis_status"] == "success"
    assert row["error_reason"] == ""
    assert (output_dir / "masks" / "sample_tif_mask.tif").exists()


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
    assert row["saved_mask_path"] == ""


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


def test_batch_mask_names_do_not_collide_across_extensions(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("sample.png", "sample.tif"):
        skio.imsave(input_dir / name, np.zeros((16, 16), dtype=np.uint8))
    monkeypatch.setattr(batch, "predict_laticifer_mask", _fake_prediction)

    rows = list(batch.run_batch_processing(
        str(input_dir), str(output_dir), run_network=False
    ))

    mask_paths = [Path(item[3]["saved_mask_path"]) for item in rows]
    assert len(set(mask_paths)) == 2
    assert all(path.exists() for path in mask_paths)


def test_create_batch_run_directory_never_reuses_existing_output(monkeypatch, tmp_path):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, _format):
            return "20260915_120000"

    monkeypatch.setattr(batch, "datetime", FixedDateTime)

    first = batch.create_batch_run_directory(str(tmp_path))
    second = batch.create_batch_run_directory(str(tmp_path))

    assert first.name == "batch_20260915_120000"
    assert second.name == "batch_20260915_120000_2"
    assert first.is_dir()
    assert second.is_dir()


def test_write_batch_csv_replaces_checkpoint_and_removes_temporary_file(tmp_path):
    run_dir = tmp_path / "run"
    first_row = {"filename": "first.tif", "analysis_status": "success"}
    second_row = {"filename": "second.tif", "analysis_status": "failed"}

    batch.write_batch_csv(str(run_dir), [first_row])
    batch.write_batch_csv(str(run_dir), [first_row, second_row])

    saved = batch.pd.read_csv(run_dir / "batch_results.csv")
    assert saved["filename"].tolist() == ["first.tif", "second.tif"]
    assert not (run_dir / "batch_results.tmp.csv").exists()


@pytest.mark.parametrize("shape", [(16, 16), (16, 16, 3), (16, 16, 4)])
def test_batch_image_validation_accepts_2d_rgb_and_rgba(shape):
    batch.validate_batch_image(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(3, 16, 16), (2, 3, 16, 16), (16,)])
def test_batch_image_validation_rejects_ambiguous_dimensions(shape):
    with pytest.raises(ValueError, match="Unsupported or ambiguous"):
        batch.validate_batch_image(np.zeros(shape, dtype=np.uint8))


def test_find_batch_images_is_case_insensitive_and_supports_jpeg(tmp_path):
    for name in ("A.TIF", "b.JPEG", "ignore.txt"):
        (tmp_path / name).touch()

    assert [path.name for path in batch.find_batch_images(str(tmp_path))] == [
        "A.TIF", "b.JPEG"
    ]
