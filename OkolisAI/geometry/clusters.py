"""DBSCAN clustering on residual points."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d


@dataclass
class ClusterSegment:
    indices: np.ndarray
    centroid: np.ndarray
    obb: o3d.geometry.OrientedBoundingBox | None


def cluster_residual(pcd: o3d.geometry.PointCloud,
                     remaining: np.ndarray,
                     eps: float = 0.15,
                     min_points: int = 20) -> list[ClusterSegment]:
    """DBSCAN on the subset `remaining` (index array into pcd).
    Returns clusters with indices expressed in the original pcd frame."""
    if len(remaining) == 0:
        return []
    sub = pcd.select_by_index(remaining)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(sub.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters: list[ClusterSegment] = []
    xyz_all = np.asarray(pcd.points)
    for lbl in range(labels.max() + 1) if labels.size else []:
        local = np.where(labels == lbl)[0]
        if len(local) < min_points:
            continue
        global_idx = remaining[local]
        pts = xyz_all[global_idx]
        try:
            obb = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pts)).get_oriented_bounding_box()
        except Exception:
            obb = None
        clusters.append(ClusterSegment(
            indices=global_idx,
            centroid=pts.mean(axis=0),
            obb=obb))
    return clusters
