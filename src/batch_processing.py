# src/batch_processing.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple, List

import numpy as np
import pandas as pd
from skimage import io as skio

from model import predict_laticifer_mask
from utils.preprocessing import apply_clahe
from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect


def run_batch_processing(
    in_dir_str: str,
    out_dir_str: str,
    *,
    num_lines: int = 10,
) -> Iterator[Tuple[int, int, str, Dict]]:
    """
    Generator that keeps the same logic as before and yields:
      (curr, total, filename, row_dict)
    """
    in_path = Path(in_dir_str)
    out_path = Path(out_dir_str)
    masks_out = out_path / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for ext in ["*.tif", "*.tiff", "*.jpg", "*.png"]:
        files.extend(list(in_path.glob(ext)))

    # Optional but nice: deterministic order (doesn't change functionality, just consistency)
    files = sorted(files)

    total = len(files)

    for i, f in enumerate(files, start=1):
        # progress
        row: Dict = {"filename": f.name}

        try:
            img = skio.imread(f)
            img_enh = apply_clahe(img)  # same as your current behavior
            mask = predict_laticifer_mask(img_enh)

            # Metrics (same)
            px_whole = analyze_density_pixel_ratio(mask)
            px_tissue = analyze_density_pixel_ratio(mask, use_tissue_mask=True)
            tr_h, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="horizontal")
            tr_v, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="vertical")
            tr_b, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="both")

            # Save mask (same)
            mask_name = f"{f.stem}_mask.tif"
            skio.imsave(masks_out / mask_name, (mask * 255).astype(np.uint8))

            row.update({
                "density_percent_whole_image": px_whole["density_percentage"],
                "density_percent_tissue_mask": px_tissue["density_percentage"],
                "transect_mean_intersections_horizontal": tr_h.get("mean_intersections_per_line", ""),
                "transect_mean_intersections_vertical": tr_v.get("mean_intersections_per_line", ""),
                "transect_mean_intersections_both": tr_b.get("mean_intersections_per_line", ""),
            })

        except Exception as e:
            # keep current behavior (print), but also return something useful
            print(f"Error on {f.name}: {e}")
            row.update({
                "density_percent_whole_image": "",
                "density_percent_tissue_mask": "",
                "transect_mean_intersections_horizontal": "",
                "transect_mean_intersections_vertical": "",
                "transect_mean_intersections_both": "",
            })

        yield (i, total, f.name, row)


def write_batch_results_csv(out_dir_str: str, results: List[Dict]) -> None:
    out_path = Path(out_dir_str)
    if results:
        pd.DataFrame(results).to_csv(out_path / "batch_results.csv", index=False)