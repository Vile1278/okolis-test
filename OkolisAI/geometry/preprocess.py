"""Denoise, downsample, normal estimation, gravity alignment."""
from __future__ import annotations
import numpy as np
import open3d as o3d


def remove_outliers(pcd: o3d.geometry.PointCloud,
                    nb_neighbors: int = 20, std_ratio: float = 2.0,
                    radius: float = 0.10, min_nb: int = 5) -> o3d.geometry.PointCloud:
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd, _ = pcd.remove_radius_outlier(nb_points=min_nb, radius=radius)
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel: float = 0.03,
                     trace: bool = False):
    """Voxel downsample. If trace=True, returns (pcd_ds, index_map) so you can
    lift per-point predictions back to the original cloud."""
    if not trace:
        return pcd.voxel_down_sample(voxel)
    min_b = pcd.get_min_bound() - voxel * 0.5
    max_b = pcd.get_max_bound() + voxel * 0.5
    pcd_ds, _, idx_list = pcd.voxel_down_sample_and_trace(voxel, min_b, max_b, False)
    return pcd_ds, idx_list


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.20, k: int = 30,
                     viewpoint: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k))
    if viewpoint is None:
        viewpoint = np.zeros(3)
    pcd.orient_normals_towards_camera_location(viewpoint)
    return pcd


def gravity_align(pcd: o3d.geometry.PointCloud,
                  up_hint: np.ndarray | None = None) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Estimate a gravity-up alignment using the dominant horizontal plane (largest
    near-vertical-normal plane via RANSAC on low-Z band). Returns (aligned_pcd, R)."""
    xyz = np.asarray(pcd.points)
    z_lo = np.quantile(xyz[:, 2], 0.05)
    z_hi = z_lo + 0.6  # 60 cm band near the bottom
    band = pcd.select_by_index(np.where((xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi))[0])
    if len(band.points) < 500:
        return pcd, np.eye(3)
    plane, _ = band.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=500)
    n = np.array(plane[:3]); n /= np.linalg.norm(n)
    target = np.array([0.0, 0.0, 1.0]) if up_hint is None else up_hint / np.linalg.norm(up_hint)
    if n @ target < 0: n = -n
    v = np.cross(n, target); s = np.linalg.norm(v); c = n @ target
    if s < 1e-8:
        return pcd, np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd, R


def preprocess(pcd: o3d.geometry.PointCloud, voxel: float = 0.03) -> o3d.geometry.PointCloud:
    pcd = remove_outliers(pcd)
    pcd, _ = gravity_align(pcd)
    pcd = voxel_downsample(pcd, voxel=voxel)
    pcd = estimate_normals(pcd, radius=max(0.15, voxel * 5), k=30)
    return pcd
