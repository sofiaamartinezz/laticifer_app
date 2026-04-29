# data/annotations.py
"""
Handles the annotations.csv log and dataset-root resolution via Qt dialogs.

Qt is intentionally confined to this module (ensure_dataset_root) and to
the save_annotation entry point. data/io.py remains Qt-free.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from qtpy.QtWidgets import QFileDialog, QMessageBox

from data.io import infer_dataset_root, save_image_and_mask
from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect


# -----------------------------------------------------------------------------
#  Dataset root resolution
# -----------------------------------------------------------------------------

def ensure_dataset_root(
    parent_widget,
    image_layer,
    dataset_root: Optional[Path],
) -> Optional[Path]:
    """
    Return a confirmed dataset root (a folder named 'dataset').

    1. If dataset_root is already known, return it immediately.
    2. Try to auto-detect from image_layer source path.
    3. Ask the user via a folder dialog; keep asking until they pick a valid
       folder or cancel.
    """
    if dataset_root is not None:
        return dataset_root

    detected = infer_dataset_root(image_layer)
    if detected is not None:
        print(f"[INFO] Auto-detected dataset root: {detected}")
        return detected

    while True:
        directory = QFileDialog.getExistingDirectory(
            parent_widget,
            "Select the dataset folder (must be named 'dataset')",
        )
        if not directory:
            return None

        candidate = Path(directory)
        p = candidate
        for _ in range(5):
            if p.name == "dataset":
                return p
            p = p.parent

        QMessageBox.warning(
            parent_widget,
            "Invalid folder",
            "The selected folder is not named 'dataset'.\nPlease try again.",
        )


# -----------------------------------------------------------------------------
#  CSV schema
# -----------------------------------------------------------------------------

_FIELDNAMES = [
    "image_path",
    "mask_path",
    "timestamp",
    "initialized_from_model",
    "image_shape_y",
    "image_shape_x",
    "laticifer_pixels",
    "density_tissue",
    "density",
    "transect_direction",
    "transect_num_lines",
    "transect_mean_intersections_per_line",
]


def _append_csv(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
#  Save annotation (main entry point called from ui)
# -----------------------------------------------------------------------------

def save_annotation(
    parent_widget,
    image_layer,
    labels_layer,
    dataset_root: Path,
    initialized_from_model: bool,
    transect_num_lines: int = 10,
    transect_direction: str = "both",
    transect_mean: Optional[float] = None,
) -> Optional[Tuple[Path, Path]]:
    """
    Save image + mask to disk and append a row to annotations.csv.

    Pixel-ratio density is always recomputed from the current mask.
    Transect density uses `transect_mean` if provided (i.e. the user already
    ran a transect calculation, possibly with edited lines) — this ensures the
    CSV reflects the exact result the user saw, not a regenerated one.
    If `transect_mean` is None, it is computed automatically from the mask.
    """
    if image_layer is None or labels_layer is None:
        QMessageBox.warning(
            parent_widget, "Missing data", "Ensure both image and mask are present."
        )
        return None

    image_data  = np.asarray(image_layer.data)
    mask_labels = np.asarray(labels_layer.data)

    base = Path(image_layer.name).stem if image_layer.name and image_layer.name != "Image" \
        else datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        image_out, mask_out = save_image_and_mask(
            image_data, mask_labels, base, dataset_root
        )
    except Exception as exc:
        QMessageBox.critical(parent_widget, "Save error", f"Failed to save files: {exc}")
        return None

    # Pixel-ratio density — always recomputed from the current mask
    px_tissue = analyze_density_pixel_ratio(mask_labels, use_tissue_mask=True)
    px_whole  = analyze_density_pixel_ratio(mask_labels)

    direction = (transect_direction or "both").lower().strip()
    if direction not in ("horizontal", "vertical", "both"):
        direction = "horizontal"
    num_lines = max(1, int(transect_num_lines))

    # Transect density — use the value the user already saw if available,
    # otherwise compute it automatically from the mask (no edited geometry)
    if transect_mean is not None and np.isfinite(transect_mean):
        tr_mean = float(transect_mean)
    else:
        tr_stats, _, _ = analyze_density_transect(
            mask_labels, num_lines=num_lines, direction=direction
        )
        tr_mean = tr_stats.get("mean_intersections_per_line", float("nan"))

    def _fmt(v) -> str:
        return f"{float(v):.6f}" if np.isfinite(float(v)) else ""

    row = {
        "image_path":           str(image_out),
        "mask_path":            str(mask_out),
        "timestamp":            datetime.now().isoformat(),
        "initialized_from_model": "True" if initialized_from_model else "False",
        "image_shape_y":        int(image_data.shape[0]) if image_data.ndim >= 2 else "",
        "image_shape_x":        int(image_data.shape[1]) if image_data.ndim >= 2 else "",
        "laticifer_pixels":     int(px_whole["laticifer_pixels"]),
        "density_tissue":       _fmt(px_tissue["pixel_ratio"]),
        "density":              _fmt(px_whole["pixel_ratio"]),
        "transect_direction":   direction,
        "transect_num_lines":   num_lines,
        "transect_mean_intersections_per_line": _fmt(tr_mean),
    }

    try:
        _append_csv(dataset_root / "annotations.csv", row)
    except Exception as exc:
        QMessageBox.critical(parent_widget, "CSV error", f"Failed to update CSV: {exc}")
        return None

    QMessageBox.information(
        parent_widget,
        "Saved",
        f"Annotation saved to:\n{mask_out}\nLog updated at {dataset_root / 'annotations.csv'}",
    )
    return image_out, mask_out