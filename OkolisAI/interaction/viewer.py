"""Minimal Open3D viewer: render scene colored by semantic class + selection."""
from __future__ import annotations
import numpy as np
import open3d as o3d

from ..scene.scene import Scene

CLASS_COLORS = {
    "unlabeled":  [0.50, 0.50, 0.50],
    "ground":     [0.60, 0.40, 0.20],
    "road":       [0.25, 0.25, 0.25],
    "sidewalk":   [0.70, 0.70, 0.70],
    "building":   [0.90, 0.20, 0.20],
    "fence":      [0.90, 0.60, 0.10],
    "vegetation": [0.10, 0.65, 0.10],
    "vehicle":    [0.20, 0.40, 0.90],
}


def colorize(scene: Scene, selected_wall_id: str | None = None) -> np.ndarray:
    n = len(scene.points)
    cols = np.tile(np.array(CLASS_COLORS["unlabeled"]), (n, 1))
    for s in scene.segments:
        cols[s.indices] = CLASS_COLORS.get(s.semantic, CLASS_COLORS["unlabeled"])
    cols[scene.synthetic_mask] = [0.1, 0.6, 1.0]  # highlight synthetic
    # Selection: brighten points of the picked wall
    if selected_wall_id:
        for w in scene.walls:
            if w.id == selected_wall_id:
                seg = next((s for s in scene.segments
                            if s.id == w.source_segment_id), None)
                if seg is not None:
                    cols[seg.indices] = [1.0, 1.0, 0.2]
    return cols


def show(scene: Scene, selected_wall_id: str | None = None, title: str = "Okoliš AI"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.points)
    pcd.colors = o3d.utility.Vector3dVector(colorize(scene, selected_wall_id))
    geoms = [pcd]
    for w in scene.walls:
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.stack([w.start, w.end])),
            lines=o3d.utility.Vector2iVector([[0, 1]]))
        line.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        geoms.append(line)
    o3d.visualization.draw_geometries(geoms, window_name=title)
