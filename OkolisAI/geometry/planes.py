"""Iterative RANSAC plane extraction — wall candidates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d


@dataclass
class PlaneSegment:
    indices: np.ndarray          # into the input pcd
    plane: np.ndarray            # [a,b,c,d]  ax+by+cz+d=0
    normal: np.ndarray           # unit (3,)
    area: float
    verticality: float           # 1.0 = perfectly vertical
    centroid: np.ndarray


def _plane_area(xyz: np.ndarray, normal: np.ndarray) -> float:
    """Area via projection onto the plane's principal axes (2D bbox of projection)."""
    c = xyz.mean(axis=0)
    # build plane basis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p = xyz - c
    uu = p @ u; vv = p @ v
    return float((uu.max() - uu.min()) * (vv.max() - vv.min()))


def extract_planes(pcd: o3d.geometry.PointCloud,
                   exclude: np.ndarray | None = None,
                   distance: float = 0.03,
                   min_inliers: int = 500,
                   max_planes: int = 20,
                   vertical_only: bool = True,
                   verticality_threshold: float = 0.3) -> list[PlaneSegment]:
    """Iteratively fit planes. `exclude` is an index mask of points to skip
    (e.g. ground). Returns a list of PlaneSegment in discovery order."""
    xyz_all = np.asarray(pcd.points)
    n_total = len(xyz_all)
    available = np.ones(n_total, dtype=bool)
    if exclude is not None:
        available[exclude] = False

    segments: list[PlaneSegment] = []
    for _ in range(max_planes):
        idx_pool = np.where(available)[0]
        if len(idx_pool) < min_inliers:
            break
        sub = pcd.select_by_index(idx_pool)
        try:
            plane, local_inliers = sub.segment_plane(
                distance_threshold=distance, ransac_n=3, num_iterations=1000)
        except RuntimeError:
            break
        if len(local_inliers) < min_inliers:
            break
        n = np.array(plane[:3]); nn = np.linalg.norm(n)
        if nn < 1e-9: break
        n /= nn
        if vertical_only and abs(n[2]) > verticality_threshold:
            # Horizontal-ish plane: remove its points so iteration progresses,
            # but don't record it (ground or roof).
            global_idx = idx_pool[local_inliers]
            available[global_idx] = False
            continue
        global_idx = idx_pool[local_inliers]
        pts = xyz_all[global_idx]
        seg = PlaneSegment(
            indices=global_idx,
            plane=np.array(plane),
            normal=n,
            area=_plane_area(pts, n),
            verticality=1.0 - abs(n[2]),
            centroid=pts.mean(axis=0),
        )
        segments.append(seg)
        available[global_idx] = False
    return segments
