import csv

from data.annotations import _FIELDNAMES, _append_csv


def test_append_csv_migrates_old_schema(tmp_path):
    path = tmp_path / "annotations.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "density"])
        writer.writeheader()
        writer.writerow({"image_path": "old.tif", "density": "0.1"})

    new_row = {field: "" for field in _FIELDNAMES}
    new_row.update({"image_path": "new.tif", "density": "0.2", "scale_source": "manual_entry"})
    _append_csv(path, new_row)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == _FIELDNAMES
    assert rows[0]["image_path"] == "old.tif"
    assert rows[0]["scale_source"] == ""
    assert rows[1]["scale_source"] == "manual_entry"
