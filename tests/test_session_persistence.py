import json

import numpy as np
import pytest
from skimage import io as skio

from data.session import (
    SESSION_FORMAT,
    SESSION_VERSION,
    SessionData,
    SessionError,
    load_session,
    save_session,
)


def test_session_round_trip_with_portable_mask_sidecar(tmp_path):
    image_path = tmp_path / "image.tif"
    skio.imsave(image_path, np.zeros((16, 20), dtype=np.uint8))
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[:, 10] = 1
    session_path = tmp_path / "analysis.json"
    expected = SessionData(
        image_path=str(image_path),
        initialized_from_model=True,
        um_per_px=0.5,
        scale_source="reference_line",
        scale_reference_pixels=200.0,
        scale_reference_length_um=100.0,
        transect_num_lines=1,
        transect_direction="horizontal",
        transect_lines=[[[8.0, 0.0], [8.0, 19.0]]],
        active_tab=2,
    )

    save_session(session_path, expected, mask)
    restored = load_session(session_path)

    document = json.loads(session_path.read_text(encoding="utf-8"))
    assert document["format"] == SESSION_FORMAT
    assert document["version"] == SESSION_VERSION
    assert document["session"]["image_path"] == "image.tif"
    assert document["session"]["mask_path"] == "analysis_mask.tif"
    assert restored.image_path == str(image_path.resolve())
    assert restored.mask_path == str((tmp_path / "analysis_mask.tif").resolve())
    assert restored.um_per_px == pytest.approx(0.5)
    assert restored.transect_lines == expected.transect_lines


@pytest.mark.parametrize(
    "document,error",
    [
        ({"not": "json session"}, "not a LatexLens"),
        ({"format": SESSION_FORMAT, "version": 999, "session": {}}, "Unsupported"),
    ],
)
def test_session_rejects_unknown_format_or_version(tmp_path, document, error):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SessionError, match=error):
        load_session(path)


def test_session_rejects_corrupt_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(SessionError, match="Could not read"):
        load_session(path)


def test_session_checks_referenced_files_before_restore(tmp_path):
    path = tmp_path / "missing.json"
    document = {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "session": {"image_path": "missing.tif"},
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SessionError, match="Image file not found"):
        load_session(path)


def test_session_rejects_invalid_transect_geometry():
    session = SessionData(
        image_path="image.tif",
        transect_lines=[[['not-a-number', 0], [1, 2]]],
    )

    with pytest.raises(SessionError, match="invalid transect geometry"):
        session.validated()
