# dataset.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import csv
from datetime import datetime
import numpy as np
from qtpy.QtWidgets import QFileDialog, QMessageBox
from skimage import io as skio


def ensure_dataset_root(
    parent_widget,
    current_image_layer,
    dataset_root: Optional[Path],
) -> Optional[Path]:
    """
    Ensures that the dataset root is a folder literally named 'dataset'.

    Rules:
    - If an image is located under .../dataset/images/, automatically use .../dataset.
    - If the user selects a folder named 'dataset', accept it.
    - If the user selects a subfolder like 'images' or 'masks',
      move up until reaching 'dataset'.
    - If no parent folder is named 'dataset', show a warning and re-ask.
    """
    if dataset_root is not None:
        return dataset_root

    from pathlib import Path as _Path

    # 1) Try infer from image path
    img_path = None
    if current_image_layer is not None:
        src = getattr(current_image_layer, "source", None)
        if src is not None and getattr(src, "path", None):
            try:
                img_path = _Path(src.path)
            except Exception:
                img_path = None

        if img_path is None and isinstance(current_image_layer.metadata, dict):
            for key in ("source", "filename", "file_name", "file_path"):
                if key in current_image_layer.metadata:
                    try:
                        img_path = _Path(current_image_layer.metadata[key])
                        break
                    except Exception:
                        continue

        if img_path is not None:
            p = img_path
            for _ in range(5):
                if p.name == "dataset":
                    print(f"[INFO] Auto-detected dataset root: {p}")
                    return p
                p = p.parent

    # 2) Ask user
    while True:
        directory = QFileDialog.getExistingDirectory(
            parent_widget,
            "Select the dataset folder (must be named 'dataset')"
        )
        if not directory:
            return None

        candidate = _Path(directory)

        p = candidate
        climbed = False
        for _ in range(5):
            if p.name == "dataset":
                if climbed:
                    print(f"[INFO] Adjusted selected folder to dataset root: {p}")
                return p
            p = p.parent
            climbed = True

        QMessageBox.warning(
            parent_widget,
            "Invalid folder",
            "You must select a folder named 'dataset'.\n"
            "Please try again."
        )


def infer_mask_path_for_image(
    image_layer,
    dataset_root: Optional[Path],
) -> Optional[Path]:
    """
    Try to guess the mask path for a given image using:
    - dataset_root/masks/<name>_mask.tif
    - if image is in .../images/, look in sibling .../masks/
    """
    from pathlib import Path

    if image_layer is None:
        return None

    img_shape = np.asarray(image_layer.data).shape[:2]
    mask_path: Optional[Path] = None

    base_name = Path(image_layer.name).stem if image_layer.name else None
    if base_name and dataset_root is not None:
        candidate = dataset_root / "masks" / f"{base_name}_mask.tif"
        if candidate.exists():
            return candidate

    img_path: Optional[Path] = None
    src = getattr(image_layer, "source", None)
    if src is not None and getattr(src, "path", None):
        try:
            img_path = Path(src.path)
        except TypeError:
            img_path = None

    if img_path is None and isinstance(image_layer.metadata, dict):
        for key in ("source", "filename", "file_name", "file_path"):
            if key in image_layer.metadata:
                try:
                    img_path = Path(image_layer.metadata[key])
                    break
                except TypeError:
                    continue

    if img_path is not None:
        base_name = img_path.stem
        if img_path.parent.name == "images":
            root = img_path.parent.parent
            candidate = root / "masks" / f"{base_name}_mask.tif"
            if candidate.exists():
                return candidate

    return mask_path


def load_mask_from_path(mask_path: Path, expected_shape) -> Optional[np.ndarray]:
    """
    Load a mask from disk, ensure 2D, check shape, and return 0/1 uint8 labels.
    """
    try:
        mask_img = skio.imread(mask_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load mask: {exc}")
        return None

    if mask_img.ndim == 3:
        mask_img = mask_img[..., 0]

    if mask_img.shape != expected_shape:
        print(f"[WARN] Shape mismatch: mask {mask_img.shape} vs image {expected_shape}")
        return None

    mask_labels = (mask_img > 0).astype(np.uint8)
    return mask_labels


def save_annotation(
    parent_widget,
    image_layer,
    labels_layer,
    dataset_root: Path,
    initialized_from_model: bool,
    transect_num_lines: int = 10,
    transect_direction: str = "both",
) -> Optional[Tuple[Path, Path]]:
    """
    Save image, mask, and append an entry to annotations.csv.
    Computes and stores density using BOTH methods on save.
    """
    import csv
    from datetime import datetime

    from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect

    if image_layer is None or labels_layer is None:
        QMessageBox.warning(parent_widget, "Missing data", "Ensure both image and mask are present.")
        return None

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    base = image_layer.name
    if not base or base == "Image":
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base).stem

    image_out = images_dir / f"{base}.tif"
    mask_out = masks_dir / f"{base}_mask.tif"

    try:
        image_data = np.asarray(image_layer.data)
        mask_labels = np.asarray(labels_layer.data)
        mask_binary = (mask_labels > 0).astype(np.uint8) * 255

        skio.imsave(image_out, image_data)
        skio.imsave(mask_out, mask_binary)
    except Exception as exc:
        QMessageBox.critical(parent_widget, "Save error", f"Failed to save data: {exc}")
        return None

    # Quantification
    px_stats = analyze_density_pixel_ratio(mask_labels)

    direction = (transect_direction or "both").lower().strip()
    if direction not in ("horizontal", "vertical", "both"):
        direction = "horizontal"
    num_lines = max(1, int(transect_num_lines))

    tr_stats, _lines, _pts = analyze_density_transect(
        mask_labels,
        num_lines=num_lines,
        direction=direction,
        roi=None,
        start_frac=0.1,
        end_frac=0.9,
    )

    # CSV 
    csv_path = dataset_root / "annotations.csv"

    FIELDNAMES = [
        "image_path",
        "mask_path",
        "timestamp",
        "initialized_from_model",
        "image_shape_y",
        "image_shape_x",
        "laticifer_pixels",
        "density",
        "transect_direction",
        "transect_num_lines",
        "transect_mean_intersections_per_line",
    ]

    image_shape_y = int(image_data.shape[0]) if image_data.ndim >= 2 else ""
    image_shape_x = int(image_data.shape[1]) if image_data.ndim >= 2 else ""

    row = {
        "image_path": str(image_out),
        "mask_path": str(mask_out),
        "timestamp": datetime.now().isoformat(),
        "initialized_from_model": "True" if initialized_from_model else "False",
        "image_shape_y": image_shape_y,
        "image_shape_x": image_shape_x,
        "laticifer_pixels": int(px_stats["laticifer_pixels"]),
        "density": (
            f"{float(px_stats['pixel_ratio']):.6f}"
            if np.isfinite(px_stats["pixel_ratio"]) else ""
        ),
        "transect_direction": direction,
        "transect_num_lines": num_lines,
        "transect_mean_intersections_per_line": (
            f"{float(tr_stats.get('mean_intersections_per_line', float('nan'))):.6f}"
            if np.isfinite(tr_stats.get("mean_intersections_per_line", float("nan"))) else ""
        ),
    }

    write_header = not csv_path.exists()

    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        QMessageBox.critical(parent_widget, "CSV error", f"Failed to update CSV: {exc}")
        return None

    QMessageBox.information(
        parent_widget,
        "Saved",
        f"Annotation saved to:\n{mask_out}\nLog updated at {csv_path}",
    )
    return image_out, mask_out

