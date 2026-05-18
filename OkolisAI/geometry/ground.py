"""Ground extraction: primary = lightweight CSF-style; fallback = RANSAC plane.

This is a pragmatic CSF substitute (no external dep). For production-grade CSF,
swap in `CSF` (pip install cloth-simulation-filter) behind the same interface."""
from __future__ import annotations
import numpy as np
import open3d as o3d


def extract_ground_ransac(pcd: o3d.geometry.PointCloud,
                          distance: float = 0.05) -> np.ndarray:
    """Return boolean mask of ground points using best horizontal plane."""
    xyz = np.asarray(pcd.points)
    plane, inliers = pcd.segment_plane(distance_threshold=distance,
                                       ransac_n=3, num_iterations=1000)
    n = np.array(plane[:3]); n /= np.linalg.norm(n)
    if abs(n[2]) < 0.8:
        # Best plane isn't horizontal enough — no single ground plane.
        return np.zeros(len(xyz), dtype=bool)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[inliers] = True
    return mask


def extract_ground_grid(pcd: o3d.geometry.PointCloud,
                        cell: float = 0.3, z_tol: float = 0.15) -> np.ndarray:
    """Grid-based lowest-point ground: handles sloped outdoor terrain.

    Divide XY into cells, take the min-Z per cell as the local ground height,
    then mark points within z_tol of that local floor as ground. Works where a
    single RANSAC plane fails (slopes, berms, terraces)."""
    xyz = np.asarray(pcd.points)
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    ix = np.floor((xyz[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    order = np.argsort(key)
    sorted_key = key[order]
    sorted_z = xyz[order, 2]
    # unique cell boundaries
    uniq, starts = np.unique(sorted_key, return_index=True)
    ends = np.r_[starts[1:], len(sorted_key)]
    cell_min_z = {k: sorted_z[s:e].min() for k, s, e in zip(uniq, starts, ends)}
    # Smooth: replace each cell's min with min over 3x3 neighborhood to kill
    # false "islands" floating above the true ground (cars, bushes bottoms).
    max_iy = iy.max() + 1
    smoothed = {}
    for k in cell_min_z:
        cx, cy = divmod(k, max_iy)
        vals = [cell_min_z[(cx + dx) * max_iy + (cy + dy)]
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (cx + dx) * max_iy + (cy + dy) in cell_min_z]
        smoothed[k] = float(np.median(vals))
    local_floor = np.array([smoothed[k] for k in key])
    return xyz[:, 2] <= local_floor + z_tol


def extract_ground(pcd: o3d.geometry.PointCloud,
                   method: str = "grid", **kw) -> np.ndarray:
    if method == "grid":
        return extract_ground_grid(pcd, **kw)
    if method == "ransac":
        return extract_ground_ransac(pcd, **kw)
    raise ValueError(method)


def height_above_ground(pcd: o3d.geometry.PointCloud,
                        ground_mask: np.ndarray, cell: float = 0.3) -> np.ndarray:
    """Per-point height above the nearest-in-XY ground point. Used as ML feature."""
    xyz = np.asarray(pcd.points)
    g = xyz[ground_mask]
    if len(g) == 0:
        return xyz[:, 2] - xyz[:, 2].min()
    # Grid lookup: for each (x,y) cell, store median z of ground points.
    # IMPORTANT: compute max_iy from ALL points (not just ground) to avoid
    # key collisions when non-ground points extend beyond ground Y range.
    all_iy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    max_iy = int(all_iy.max()) + 1 if len(all_iy) else 1
    ix = np.floor((g[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    iy = np.floor((g[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    keys = ix * max_iy + iy
    table: dict[int, list] = {}
    for k, z in zip(keys, g[:, 2]):
        table.setdefault(int(k), []).append(float(z))
    med = {k: float(np.median(v)) for k, v in table.items()}
    qx = np.floor((xyz[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    qy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    qk = qx * max_iy + qy
    ground_z = np.array([med.get(int(k), np.nan) for k in qk])
    # Fill NaN with global min ground z.
    if np.isnan(ground_z).any():
        fallback = float(np.nanmin(ground_z)) if np.isfinite(np.nanmin(ground_z)) else g[:, 2].min()
        ground_z[np.isnan(ground_z)] = fallback
    return xyz[:, 2] - ground_z
