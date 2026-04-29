# data/batch.py
"""
Batch processing: runs prediction + density metrics over a folder of images
and writes a CSV summary. No Qt dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from skimage import io as skio

from model.predictor import predict_laticifer_mask
from utils.preprocessing import apply_clahe
from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect


_IMAGE_EXTENSIONS = ["*.tif", "*.tiff", "*.jpg", "*.png"]


def run_batch_processing(
    in_dir_str: str,
    out_dir_str: str,
    *,
    num_lines: int = 10,
) -> Iterator[Tuple[int, int, str, Dict]]:
    """
    Process every image in in_dir_str: predict mask, compute metrics, save mask.

    Yields (current_index, total, filename, result_row) for progress reporting.
    The caller is responsible for collecting rows and writing the CSV.
    """
    in_path  = Path(in_dir_str)
    out_path = Path(out_dir_str)
    masks_out = out_path / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(
        f for ext in _IMAGE_EXTENSIONS for f in in_path.glob(ext)
    )
    total = len(files)

    for i, f in enumerate(files, start=1):
        row: Dict = {"filename": f.name}
        try:
            img     = skio.imread(f)
            img_enh = apply_clahe(img)
            mask    = predict_laticifer_mask(img_enh)

            px_whole  = analyze_density_pixel_ratio(mask)
            px_tissue = analyze_density_pixel_ratio(mask, use_tissue_mask=True)
            tr_h, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="horizontal")
            tr_v, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="vertical")
            tr_b, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="both")

            skio.imsave(masks_out / f"{f.stem}_mask.tif", (mask * 255).astype(np.uint8))

            row.update({
                "density_percent_whole_image":              px_whole["density_percentage"],
                "density_percent_tissue_mask":              px_tissue["density_percentage"],
                "transect_mean_intersections_horizontal":   tr_h.get("mean_intersections_per_line", ""),
                "transect_mean_intersections_vertical":     tr_v.get("mean_intersections_per_line", ""),
                "transect_mean_intersections_both":         tr_b.get("mean_intersections_per_line", ""),
            })
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            row.update({
                "density_percent_whole_image":              "",
                "density_percent_tissue_mask":              "",
                "transect_mean_intersections_horizontal":   "",
                "transect_mean_intersections_vertical":     "",
                "transect_mean_intersections_both":         "",
            })

        yield i, total, f.name, row


def write_batch_csv(out_dir_str: str, results: List[Dict]) -> None:
    """Write accumulated result rows to batch_results.csv."""
    if results:
        pd.DataFrame(results).to_csv(
            Path(out_dir_str) / "batch_results.csv", index=False
        )