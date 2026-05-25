# utils/network_analysis.py
"""
Skeleton-based network analysis for laticifer masks.

Computes:
  - Total skeleton length
  - Bifurcation (branch-point) detection and count
  - Bifurcation angles  (cluster-aware: multiple adjacent bif-pixels → one node)
  - Branch extraction and length distribution
  - Laticifer diameter via distance transform
  - Connectivity metrics (branch/node ratio, etc.)

Pure numpy / scipy / skimage — no Qt, no napari.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label


# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------

@dataclass
class NetworkStats:
    """All scalar metrics returned by run_network_analysis()."""
    # Expansion
    total_skeleton_length_px: float = 0.0
    total_skeleton_length_um: Optional[float] = None  # set when scale is known

    # Branching
    bifurcation_count: int = 0          # logical nodes (after clustering)
    endpoint_count: int = 0
    branch_count: int = 0
    mean_branch_length_px: float = float("nan")
    std_branch_length_px: float = float("nan")
    mean_bifurcation_angle_deg: float = float("nan")
    std_bifurcation_angle_deg: float = float("nan")

    # Thickness (distance-transform diameter)
    mean_diameter_px: float = float("nan")
    std_diameter_px: float = float("nan")
    median_diameter_px: float = float("nan")

    # Connectivity
    branch_node_ratio: float = float("nan")
    connected_components: int = 0

    # Scale (stored for downstream display)
    um_per_px: Optional[float] = None

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class NetworkGeometry:
    """Per-element geometric arrays for napari visualisation."""
    bifurcation_points: np.ndarray  = field(default_factory=lambda: np.zeros((0, 2)))
    endpoint_points: np.ndarray     = field(default_factory=lambda: np.zeros((0, 2)))
    branch_lines: List[np.ndarray]  = field(default_factory=list)
    diameter_map: np.ndarray        = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    branch_lengths_px: np.ndarray   = field(default_factory=lambda: np.zeros(0))
    bifurcation_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return boolean skeleton of a binary mask."""
    return skeletonize(mask > 0)


def _neighbor_count_map(skel: np.ndarray) -> np.ndarray:
    """
    For each skeleton pixel, count 8-connected skeleton neighbours.
    Returns an integer array of the same shape.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    conv = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return conv * skel.astype(np.uint8)


def _classify_pixels(skel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify skeleton pixels into bifurcation points (≥3 neighbours)
    and endpoints (exactly 1 neighbour).

    Returns:
        bif_yx : (N, 2) int array of branch-point pixel coordinates [y, x]
        end_yx : (M, 2) int array of tip pixel coordinates [y, x]
    """
    nc = _neighbor_count_map(skel)
    bif_yx = np.column_stack(np.where(nc >= 3)) if np.any(nc >= 3) else np.zeros((0, 2), dtype=int)
    end_yx = np.column_stack(np.where(nc == 1)) if np.any(nc == 1) else np.zeros((0, 2), dtype=int)
    return bif_yx, end_yx


def _cluster_bifurcation_nodes(bif_yx: np.ndarray) -> List[np.ndarray]:
    """
    Group adjacent bifurcation pixels into logical nodes via connected-component
    labelling on a small binary image.  Returns a list where each element is
    the (N_i, 2) pixel array belonging to one logical node.

    This is critical because a thick skeleton often produces a small cluster of
    3–4 adjacent pixels all classified as bifurcations; treating them as one
    node gives correct branch counts and angles.
    """
    if bif_yx.shape[0] == 0:
        return []

    # Build a minimal bounding box image to label the clusters
    y_min, x_min = bif_yx.min(axis=0)
    y_max, x_max = bif_yx.max(axis=0)
    h = y_max - y_min + 3   # +1 padding on each side
    w = x_max - x_min + 3
    img = np.zeros((h, w), dtype=np.uint8)
    img[bif_yx[:, 0] - y_min + 1, bif_yx[:, 1] - x_min + 1] = 1

    labeled, n = sk_label(img, connectivity=2, return_num=True)

    clusters: List[np.ndarray] = []
    for lbl in range(1, n + 1):
        local_yx = np.column_stack(np.where(labeled == lbl))
        # Convert back to global coords
        global_yx = local_yx + np.array([y_min - 1, x_min - 1])
        clusters.append(global_yx)
    return clusters


def _extract_branches(skel: np.ndarray, bif_yx: np.ndarray) -> List[np.ndarray]:
    """
    Trace individual branch segments by temporarily removing all bifurcation
    pixels, then labelling connected components of the remaining skeleton.

    Returns a list of (N_i, 2) yx float arrays, one per branch.
    """
    skel_no_bif = skel.copy()
    if bif_yx.shape[0] > 0:
        skel_no_bif[bif_yx[:, 0], bif_yx[:, 1]] = False

    labeled, n = sk_label(skel_no_bif, connectivity=2, return_num=True)
    branches: List[np.ndarray] = []
    for lbl in range(1, n + 1):
        yx = np.column_stack(np.where(labeled == lbl)).astype(float)
        if len(yx) > 0:
            branches.append(yx)
    return branches


def _branch_length(branch_yx: np.ndarray) -> float:
    """
    Approximate branch length by a greedy nearest-neighbour walk and summing
    Euclidean step distances between consecutive pixels.
    """
    if len(branch_yx) < 2:
        return float(len(branch_yx))
    pts = branch_yx.copy()
    ordered = [pts[0]]
    remaining = list(range(1, len(pts)))
    while remaining:
        last = ordered[-1]
        dists = np.linalg.norm(pts[remaining] - last, axis=1)
        nearest = remaining[int(np.argmin(dists))]
        ordered.append(pts[nearest])
        remaining.remove(nearest)
    ordered_arr = np.array(ordered)
    return float(np.sum(np.linalg.norm(np.diff(ordered_arr, axis=0), axis=1)))


def _angle_at_node(
    node_pixels: np.ndarray,
    skel_no_bif: np.ndarray,
    skel_full: np.ndarray,
    look_ahead: int,
) -> Optional[float]:
    """
    Estimate the opening angle at one logical bifurcation node.

    Strategy:
      1. Build a set of all pixels in this node.
      2. Find all 8-connected skel_no_bif pixels neighbouring any node pixel
         (these are the branch entry points).
      3. From each entry point walk `look_ahead` steps along the skeleton
         (not re-entering the node) to get a direction vector.
      4. Return the largest angle between any pair of direction vectors.
    """
    node_set = set(map(tuple, node_pixels.tolist()))
    H, W = skel_full.shape

    # Collect entry points: non-node skeleton pixels adjacent to the node
    entry_points = set()
    for y, x in node_pixels:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if skel_no_bif[ny, nx] and (ny, nx) not in node_set:
                        entry_points.add((ny, nx))

    if len(entry_points) < 2:
        return None

    # Compute a centroid for the node to use as walk origin
    centroid = node_pixels.mean(axis=0)

    # Walk from each entry point to get direction vectors
    vecs: List[np.ndarray] = []
    for ey, ex in entry_points:
        path = [(ey, ex)]
        visited = set(node_set)
        visited.add((ey, ex))
        cy, cx = ey, ex
        for _ in range(look_ahead - 1):
            best = None
            for ddy in (-1, 0, 1):
                for ddx in (-1, 0, 1):
                    if ddy == 0 and ddx == 0:
                        continue
                    cand = (cy + ddy, cx + ddx)
                    if cand in visited:
                        continue
                    r, c = cand
                    if 0 <= r < H and 0 <= c < W and skel_full[r, c]:
                        best = cand
            if best is None:
                break
            path.append(best)
            visited.add(best)
            cy, cx = best

        if len(path) >= 1:
            tip = np.array(path[-1], dtype=float)
            v = tip - centroid
            norm = np.linalg.norm(v)
            if norm > 0:
                vecs.append(v / norm)

    if len(vecs) < 2:
        return None

    # Largest angle between any two outgoing direction vectors
    max_angle = 0.0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos_val = float(np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cos_val)))
            if ang > max_angle:
                max_angle = ang
    return max_angle


def _compute_angles(
    clusters: List[np.ndarray],
    bif_yx: np.ndarray,
    skel: np.ndarray,
    look_ahead: int,
) -> List[float]:
    """
    Compute one bifurcation angle per logical node (cluster).
    """
    if not clusters or bif_yx.shape[0] == 0:
        return []

    # Build skel_no_bif once for all nodes
    skel_no_bif = skel.copy()
    skel_no_bif[bif_yx[:, 0], bif_yx[:, 1]] = False

    angles: List[float] = []
    for node_pixels in clusters:
        ang = _angle_at_node(node_pixels, skel_no_bif, skel, look_ahead)
        if ang is not None:
            angles.append(ang)
    return angles


def _diameter_map(mask: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """
    Distance-transform diameter map masked to skeleton pixels.
    Value at each skeleton pixel ≈ local diameter (2 × local radius).
    """
    dt = ndi.distance_transform_edt(mask > 0)
    return (dt * skel.astype(float) * 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def run_network_analysis(
    mask: np.ndarray,
    um_per_px: Optional[float] = None,
    look_ahead: int = 12,
) -> Tuple[NetworkStats, NetworkGeometry]:
    """
    Full skeleton-based network analysis of a binary laticifer mask.

    Args:
        mask      : Binary label mask (values > 0 are laticifers).
        um_per_px : Optional pixel→micron scale factor for real-unit output.
        look_ahead: Pixels to trace per branch arm when computing angles.

    Returns:
        stats    : NetworkStats dataclass with all scalar metrics.
        geometry : NetworkGeometry with arrays for napari visualisation.
    """
    stats = NetworkStats(um_per_px=um_per_px)
    geom  = NetworkGeometry()

    if mask.size == 0 or not np.any(mask > 0):
        return stats, geom

    skel = _skeleton_from_mask(mask)
    geom.diameter_map = _diameter_map(mask, skel)

    # ── Expansion ────────────────────────────────────────────────────────────
    stats.total_skeleton_length_px = float(np.count_nonzero(skel))
    if um_per_px is not None and um_per_px > 0:
        stats.total_skeleton_length_um = stats.total_skeleton_length_px * um_per_px

    # ── Node classification ───────────────────────────────────────────────────
    bif_yx, end_yx = _classify_pixels(skel)
    stats.endpoint_count = int(len(end_yx))
    geom.endpoint_points = end_yx.astype(float)

    # Cluster adjacent bifurcation pixels → logical nodes
    clusters = _cluster_bifurcation_nodes(bif_yx)
    stats.bifurcation_count = len(clusters)

    # Centroid of each cluster → display point in napari
    if clusters:
        centroids = np.array([c.mean(axis=0) for c in clusters], dtype=float)
    else:
        centroids = np.zeros((0, 2), dtype=float)
    geom.bifurcation_points = centroids

    # ── Branches ─────────────────────────────────────────────────────────────
    branches = _extract_branches(skel, bif_yx)
    geom.branch_lines = branches
    lengths = np.array([_branch_length(b) for b in branches], dtype=float)
    geom.branch_lengths_px = lengths
    stats.branch_count = len(branches)
    if len(lengths) > 0:
        stats.mean_branch_length_px = float(np.mean(lengths))
        stats.std_branch_length_px  = float(np.std(lengths))

    # ── Bifurcation angles ────────────────────────────────────────────────────
    angles = _compute_angles(clusters, bif_yx, skel, look_ahead)
    angles_arr = np.array(angles, dtype=float)
    geom.bifurcation_angles_deg = angles_arr
    if len(angles_arr) > 0:
        stats.mean_bifurcation_angle_deg = float(np.mean(angles_arr))
        stats.std_bifurcation_angle_deg  = float(np.std(angles_arr))

    # ── Diameter ─────────────────────────────────────────────────────────────
    diam_vals = geom.diameter_map[skel]
    diam_vals = diam_vals[diam_vals > 0]
    if len(diam_vals) > 0:
        stats.mean_diameter_px   = float(np.mean(diam_vals))
        stats.std_diameter_px    = float(np.std(diam_vals))
        stats.median_diameter_px = float(np.median(diam_vals))

    # ── Connectivity ──────────────────────────────────────────────────────────
    n_nodes = stats.bifurcation_count + stats.endpoint_count
    if n_nodes > 0:
        stats.branch_node_ratio = stats.branch_count / n_nodes
    _, stats.connected_components = sk_label(skel, connectivity=2, return_num=True)

    return stats, geom