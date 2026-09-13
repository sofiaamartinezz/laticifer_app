import numpy as np
import pytest
import tifffile

import data.batch as batch


def _fake_prediction(image, device=None):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[:, image.shape[1] // 2] = 1
    return mask


def test_batch_uses_each_images_metadata(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    path = input_dir / "sample.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((16, 16), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )
    monkeypatch.setattr(batch, "predict_laticifer_mask", _fake_prediction)

    results = list(
        batch.run_batch_processing(
            str(input_dir),
            str(output_dir),
            run_network=False,
            read_scale_from_metadata=True,
        )
    )

    assert len(results) == 1
    row = results[0][3]
    assert float(row["um_per_px"]) == pytest.approx(0.5)
    assert row["scale_source"] == "ome_metadata"
    assert (output_dir / "masks" / "sample.ome_mask.tif").exists()


def test_batch_marks_missing_metadata_without_reusing_manual_scale(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "plain.tif", np.zeros((16, 16), dtype=np.uint8))
    monkeypatch.setattr(batch, "predict_laticifer_mask", _fake_prediction)

    row = list(
        batch.run_batch_processing(
            str(input_dir),
            str(output_dir),
            run_network=False,
            um_per_px=9.0,
            read_scale_from_metadata=True,
        )
    )[0][3]

    assert row["um_per_px"] == ""
    assert row["scale_source"] == "metadata_unavailable"
