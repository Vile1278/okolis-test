"""PLY I/O with graceful handling of missing colors/normals."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import open3d as o3d


def load_ply(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a .ply into an Open3D point cloud. Fails loudly on empty input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Loaded point cloud is empty: {path}")
    return pcd


def save_ply(pcd: o3d.geometry.PointCloud, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)
    if not ok:
        raise IOError(f"Failed to write {path}")


def to_numpy(pcd: o3d.geometry.PointCloud) -> dict:
    return {
        "xyz": np.asarray(pcd.points, dtype=np.float64),
        "rgb": np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None,
        "normals": np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None,
    }


def from_numpy(xyz: np.ndarray, rgb: Optional[np.ndarray] = None,
               normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return pcd
