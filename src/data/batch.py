# data/batch.py
"""
Batch processing: runs prediction + density + network metrics over a folder
of images and writes a CSV summary. No Qt dependency.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from skimage import io as skio

from model.predictor import predict_laticifer_mask
from data.provenance import APP_VERSION, analysis_timestamp, measurement_units
from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect
from utils.network_analysis import run_network_analysis


_IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Column order for the output CSV
_COLUMNS = [
    "filename",
    "source_image_path",
    "saved_mask_path",
    "analysis_timestamp",
    "app_version",
    "image_shape_y",
    "image_shape_x",
    "transect_num_lines_per_direction",
    "network_analysis_enabled",
    "measurement_system",
    "length_unit",
    "area_unit",
    "analysis_status",
    "error_reason",
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
    "scale_source",
]


def _empty_row(filename: str) -> Dict:
    """Return a row with all numeric fields blank (used on error)."""
    row: Dict = {"filename": filename}
    for col in _COLUMNS[1:]:
        row[col] = ""
    return row


def create_batch_run_directory(output_root_str: str) -> Path:
    """Create and return a unique directory for one batch execution."""
    output_root = Path(output_root_str)
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = output_root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def _mask_paths(files: List[Path], masks_out: Path) -> Dict[Path, Path]:
    """Return deterministic, collision-free output paths for input images."""
    paths: Dict[Path, Path] = {}
    used_names = set()
    for image_path in files:
        extension = image_path.suffix.lstrip(".").lower() or "image"
        base_name = f"{image_path.stem}_{extension}_mask"
        candidate = f"{base_name}.tif"
        suffix = 2
        while candidate.casefold() in used_names:
            candidate = f"{base_name}_{suffix}.tif"
            suffix += 1
        used_names.add(candidate.casefold())
        paths[image_path] = masks_out / candidate
    return paths


def find_batch_images(input_dir_str: str) -> List[Path]:
    """Return supported top-level images with case-insensitive extensions."""
    input_dir = Path(input_dir_str)
    if not input_dir.is_dir():
        return []
    return sorted(
        (path for path in input_dir.iterdir()
         if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )


def validate_batch_image(image: np.ndarray) -> None:
    """Reject arrays whose channels cannot be interpreted unambiguously."""
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return
    raise ValueError(
        "Unsupported or ambiguous image dimensions "
        f"{tuple(image.shape)}. Expected a 2D grayscale image or an RGB/RGBA "
        "image with channels in the last dimension; Z-stacks and other "
        "multidimensional images must be converted before batch processing."
    )


def run_batch_processing(
    in_dir_str: str,
    out_dir_str: str,
    *,
    num_lines: int = 10,
    run_network: bool = True,
    um_per_px: Optional[float] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
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
        should_cancel: Optional callback checked between images.
    """
    in_path   = Path(in_dir_str)
    out_path  = Path(out_dir_str)
    masks_out = out_path / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    files = find_batch_images(str(in_path))
    total = len(files)
    run_timestamp = analysis_timestamp()
    mask_paths = _mask_paths(files, masks_out)

    # Convenience: convert a px value to µm, or return "" if scale unknown
    def _to_um(px_val, scale: Optional[float]) -> str:
        if scale and scale > 0:
            try:
                import math
                v = float(px_val)
                return f"{v * scale:.6f}" if math.isfinite(v) else ""
            except (TypeError, ValueError):
                return ""
        return ""

    for i, f in enumerate(files, start=1):
        if should_cancel is not None and should_cancel():
            break
        local_scale = um_per_px
        scale_source = "batch_manual" if um_per_px else "pixels_only"
        measurement_system, length_unit, area_unit = measurement_units(local_scale)
        mask_path = mask_paths[f]
        row: Dict = {
            "filename": f.name,
            "source_image_path": str(f.resolve()),
            "saved_mask_path": "",
            "analysis_timestamp": run_timestamp,
            "app_version": APP_VERSION,
            "transect_num_lines_per_direction": max(1, int(num_lines)),
            "network_analysis_enabled": bool(run_network),
            "measurement_system": measurement_system,
            "length_unit": length_unit,
            "area_unit": area_unit,
            "analysis_status": "success",
            "error_reason": "",
            "um_per_px": _fmt(local_scale) if local_scale else "",
            "scale_source": scale_source,
        }
        try:
            img  = skio.imread(f)
            validate_batch_image(img)
            row["image_shape_y"] = int(img.shape[0]) if img.ndim >= 2 else ""
            row["image_shape_x"] = int(img.shape[1]) if img.ndim >= 2 else ""
            mask = predict_laticifer_mask(img)

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
                    net_stats, _ = run_network_analysis(mask, um_per_px=local_scale)
                    row.update({
                        "skeleton_length_px":         net_stats.total_skeleton_length_px,
                        "skeleton_length_um":         _fmt(net_stats.total_skeleton_length_um) if net_stats.total_skeleton_length_um is not None else "",
                        "connected_components":       net_stats.connected_components,
                        "bifurcation_count":          net_stats.bifurcation_count,
                        "endpoint_count":             net_stats.endpoint_count,
                        "branch_count":               net_stats.branch_count,
                        "mean_branch_length_px":      _fmt(net_stats.mean_branch_length_px),
                        "mean_branch_length_um":      _to_um(net_stats.mean_branch_length_px, local_scale),
                        "std_branch_length_px":       _fmt(net_stats.std_branch_length_px),
                        "std_branch_length_um":       _to_um(net_stats.std_branch_length_px, local_scale),
                        "mean_bifurcation_angle_deg": _fmt(net_stats.mean_bifurcation_angle_deg),
                        "std_bifurcation_angle_deg":  _fmt(net_stats.std_bifurcation_angle_deg),
                        "mean_diameter_px":           _fmt(net_stats.mean_diameter_px),
                        "mean_diameter_um":           _to_um(net_stats.mean_diameter_px, local_scale),
                        "std_diameter_px":            _fmt(net_stats.std_diameter_px),
                        "std_diameter_um":            _to_um(net_stats.std_diameter_px, local_scale),
                        "median_diameter_px":         _fmt(net_stats.median_diameter_px),
                        "median_diameter_um":         _to_um(net_stats.median_diameter_px, local_scale),
                        "branch_node_ratio":          _fmt(net_stats.branch_node_ratio),
                    })
                except Exception as net_exc:
                    print(f"[WARN] {f.name}: network analysis failed: {net_exc}")
                    row["analysis_status"] = "partial"
                    row["error_reason"] = f"Network analysis failed: {net_exc}"
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
            skio.imsave(mask_path, (mask * 255).astype(np.uint8))
            row["saved_mask_path"] = str(mask_path.resolve())

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            failed_row = _empty_row(f.name)
            for key in (
                "source_image_path", "saved_mask_path", "analysis_timestamp",
                "app_version", "transect_num_lines_per_direction",
                "network_analysis_enabled", "measurement_system",
                "length_unit", "area_unit", "um_per_px", "scale_source",
            ):
                failed_row[key] = row.get(key, "")
            failed_row["analysis_status"] = "failed"
            failed_row["error_reason"] = str(e)
            row = failed_row

        yield i, total, f.name, row


def write_batch_csv(out_dir_str: str, results: List[Dict]) -> None:
    """Atomically write accumulated result rows to batch_results.csv."""
    if not results:
        return
    df = pd.DataFrame(results)
    # Reorder to canonical column order, keeping any extra columns at the end
    ordered = [c for c in _COLUMNS if c in df.columns]
    extra   = [c for c in df.columns if c not in _COLUMNS]
    out_path = Path(out_dir_str)
    out_path.mkdir(parents=True, exist_ok=True)
    final_path = out_path / "batch_results.csv"
    temporary_path = out_path / "batch_results.tmp.csv"
    df[ordered + extra].to_csv(temporary_path, index=False)
    temporary_path.replace(final_path)


def _fmt(v) -> str:
    """Format a float for CSV; blank string if not finite."""
    try:
        f = float(v)
        import math
        return f"{f:.6f}" if math.isfinite(f) else ""
    except (TypeError, ValueError):
        return ""
