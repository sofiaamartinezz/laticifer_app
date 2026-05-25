# data/batch.py
"""
Batch processing: runs prediction + density + network metrics over a folder
of images and writes a CSV summary. No Qt dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from skimage import io as skio

from model.predictor import predict_laticifer_mask
from utils.preprocessing import apply_clahe
from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect
from utils.network_analysis import run_network_analysis


_IMAGE_EXTENSIONS = ["*.tif", "*.tiff", "*.jpg", "*.png"]

# Column order for the output CSV
_COLUMNS = [
    "filename",
    # density
    "density_percent_whole_image",
    "density_percent_tissue_mask",
    "transect_mean_intersections_horizontal",
    "transect_mean_intersections_vertical",
    "transect_mean_intersections_both",
    # network — expansion
    "skeleton_length_px",
    "skeleton_length_um",          # populated only when um_per_px is provided
    "connected_components",
    # network — branching
    "bifurcation_count",
    "endpoint_count",
    "branch_count",
    "mean_branch_length_px",
    "mean_branch_length_um",
    "std_branch_length_px",
    "std_branch_length_um",
    "mean_bifurcation_angle_deg",
    "std_bifurcation_angle_deg",
    # network — thickness
    "mean_diameter_px",
    "mean_diameter_um",
    "std_diameter_px",
    "std_diameter_um",
    "median_diameter_px",
    "median_diameter_um",
    # network — connectivity
    "branch_node_ratio",
    # scale used
    "um_per_px",
]


def _empty_row(filename: str) -> Dict:
    """Return a row with all numeric fields blank (used on error)."""
    row: Dict = {"filename": filename}
    for col in _COLUMNS[1:]:
        row[col] = ""
    return row


def run_batch_processing(
    in_dir_str: str,
    out_dir_str: str,
    *,
    num_lines: int = 10,
    run_network: bool = True,
    um_per_px: Optional[float] = None,
) -> Iterator[Tuple[int, int, str, Dict]]:
    """
    Process every image in in_dir_str: predict mask, compute density and
    (optionally) network metrics, save mask.

    Yields (current_index, total, filename, result_row) for progress reporting.
    The caller is responsible for collecting rows and writing the CSV.

    Args:
        in_dir_str:  Path to the folder containing input images.
        out_dir_str: Path to the output folder.
        num_lines:   Number of transect lines per direction.
        run_network: If True, also compute skeleton network metrics.
        um_per_px:   Optional scale factor. When provided, µm columns are
                     populated alongside the pixel columns.
    """
    in_path   = Path(in_dir_str)
    out_path  = Path(out_dir_str)
    masks_out = out_path / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(
        f for ext in _IMAGE_EXTENSIONS for f in in_path.glob(ext)
    )
    total = len(files)

    # Convenience: convert a px value to µm, or return "" if scale unknown
    def _to_um(px_val) -> str:
        if um_per_px and um_per_px > 0:
            try:
                import math
                v = float(px_val)
                return f"{v * um_per_px:.6f}" if math.isfinite(v) else ""
            except (TypeError, ValueError):
                return ""
        return ""

    for i, f in enumerate(files, start=1):
        row: Dict = {"filename": f.name, "um_per_px": _fmt(um_per_px) if um_per_px else ""}
        try:
            img     = skio.imread(f)
            img_enh = apply_clahe(img)
            mask    = predict_laticifer_mask(img_enh)

            # --- Density ---
            px_whole  = analyze_density_pixel_ratio(mask)
            px_tissue = analyze_density_pixel_ratio(mask, use_tissue_mask=True)
            tr_h, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="horizontal")
            tr_v, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="vertical")
            tr_b, _, _ = analyze_density_transect(mask, num_lines=num_lines, direction="both")

            row.update({
                "density_percent_whole_image":            px_whole["density_percentage"],
                "density_percent_tissue_mask":            px_tissue["density_percentage"],
                "transect_mean_intersections_horizontal": tr_h.get("mean_intersections_per_line", ""),
                "transect_mean_intersections_vertical":   tr_v.get("mean_intersections_per_line", ""),
                "transect_mean_intersections_both":       tr_b.get("mean_intersections_per_line", ""),
            })

            # --- Network ---
            if run_network:
                try:
                    net_stats, _ = run_network_analysis(mask, um_per_px=um_per_px)
                    row.update({
                        "skeleton_length_px":         net_stats.total_skeleton_length_px,
                        "skeleton_length_um":         _fmt(net_stats.total_skeleton_length_um) if net_stats.total_skeleton_length_um is not None else "",
                        "connected_components":       net_stats.connected_components,
                        "bifurcation_count":          net_stats.bifurcation_count,
                        "endpoint_count":             net_stats.endpoint_count,
                        "branch_count":               net_stats.branch_count,
                        "mean_branch_length_px":      _fmt(net_stats.mean_branch_length_px),
                        "mean_branch_length_um":      _to_um(net_stats.mean_branch_length_px),
                        "std_branch_length_px":       _fmt(net_stats.std_branch_length_px),
                        "std_branch_length_um":       _to_um(net_stats.std_branch_length_px),
                        "mean_bifurcation_angle_deg": _fmt(net_stats.mean_bifurcation_angle_deg),
                        "std_bifurcation_angle_deg":  _fmt(net_stats.std_bifurcation_angle_deg),
                        "mean_diameter_px":           _fmt(net_stats.mean_diameter_px),
                        "mean_diameter_um":           _to_um(net_stats.mean_diameter_px),
                        "std_diameter_px":            _fmt(net_stats.std_diameter_px),
                        "std_diameter_um":            _to_um(net_stats.std_diameter_px),
                        "median_diameter_px":         _fmt(net_stats.median_diameter_px),
                        "median_diameter_um":         _to_um(net_stats.median_diameter_px),
                        "branch_node_ratio":          _fmt(net_stats.branch_node_ratio),
                    })
                except Exception as net_exc:
                    print(f"[WARN] {f.name}: network analysis failed: {net_exc}")
                    for col in [
                        "skeleton_length_px", "skeleton_length_um",
                        "connected_components",
                        "bifurcation_count", "endpoint_count", "branch_count",
                        "mean_branch_length_px", "mean_branch_length_um",
                        "std_branch_length_px", "std_branch_length_um",
                        "mean_bifurcation_angle_deg", "std_bifurcation_angle_deg",
                        "mean_diameter_px", "mean_diameter_um",
                        "std_diameter_px", "std_diameter_um",
                        "median_diameter_px", "median_diameter_um",
                        "branch_node_ratio",
                    ]:
                        row.setdefault(col, "")
            else:
                for col in [
                    "skeleton_length_px", "skeleton_length_um",
                    "connected_components",
                    "bifurcation_count", "endpoint_count", "branch_count",
                    "mean_branch_length_px", "mean_branch_length_um",
                    "std_branch_length_px", "std_branch_length_um",
                    "mean_bifurcation_angle_deg", "std_bifurcation_angle_deg",
                    "mean_diameter_px", "mean_diameter_um",
                    "std_diameter_px", "std_diameter_um",
                    "median_diameter_px", "median_diameter_um",
                    "branch_node_ratio",
                ]:
                    row[col] = ""

            # --- Save mask ---
            skio.imsave(masks_out / f"{f.stem}_mask.tif", (mask * 255).astype(np.uint8))

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            row = _empty_row(f.name)

        yield i, total, f.name, row


def write_batch_csv(out_dir_str: str, results: List[Dict]) -> None:
    """Write accumulated result rows to batch_results.csv."""
    if not results:
        return
    df = pd.DataFrame(results)
    # Reorder to canonical column order, keeping any extra columns at the end
    ordered = [c for c in _COLUMNS if c in df.columns]
    extra   = [c for c in df.columns if c not in _COLUMNS]
    df[ordered + extra].to_csv(
        Path(out_dir_str) / "batch_results.csv", index=False
    )


def _fmt(v) -> str:
    """Format a float for CSV; blank string if not finite."""
    try:
        f = float(v)
        import math
        return f"{f:.6f}" if math.isfinite(f) else ""
    except (TypeError, ValueError):
        return ""