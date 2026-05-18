"""Editing operations: pure (Scene, params) -> Scene."""
from __future__ import annotations
import numpy as np
import open3d as o3d

from ..scene.scene import Scene
from ..walls.wall_extractor import WallObject


def _box_points(center: np.ndarray, length: float, height: float, thickness: float,
                dir_vec: np.ndarray, normal: np.ndarray,
                density_m: float = 0.03) -> np.ndarray:
    """Sample points on the surface of an extruded wall box."""
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    # orthogonalize normal to direction and up
    n = normal - dir_vec * (normal @ dir_vec)
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(dir_vec, up)
    n /= np.linalg.norm(n) + 1e-12

    nl = max(int(length / density_m), 2)
    nh = max(int(height / density_m), 2)
    nt = max(int(thickness / density_m), 1)
    ls = np.linspace(-length / 2, length / 2, nl)
    hs = np.linspace(0.0, height, nh)
    ts = np.linspace(-thickness / 2, thickness / 2, nt)
    pts = []
    # two large faces (±n)
    for t_off in (-thickness / 2, thickness / 2):
        ll, hh = np.meshgrid(ls, hs, indexing="ij")
        face = (center[None, None, :]
                + dir_vec * ll[..., None]
                + up * hh[..., None]
                + n * t_off)
        pts.append(face.reshape(-1, 3))
    # two end caps (±dir)
    for l_off in (-length / 2, length / 2):
        hh, tt = np.meshgrid(hs, ts, indexing="ij")
        face = (center[None, None, :]
                + dir_vec * l_off
                + up * hh[..., None]
                + n * tt[..., None])
        pts.append(face.reshape(-1, 3))
    return np.concatenate(pts, axis=0)


def extend_wall(scene: Scene, wall_id: str, delta_length: float,
                end: str = "positive", density_m: float = 0.03) -> Scene:
    """Extend a wall along its direction vector. `end` is "positive" or "negative"."""
    idx = next((i for i, w in enumerate(scene.walls) if w.id == wall_id), -1)
    if idx < 0:
        raise KeyError(f"Unknown wall id: {wall_id}")
    wall = scene.walls[idx]
    d = wall.direction
    if end == "positive":
        new_end = wall.end + d * delta_length
        segment_center = (wall.end + new_end) / 2
        old_start, old_end = wall.start, new_end
    elif end == "negative":
        new_start = wall.start - d * delta_length
        segment_center = (wall.start + new_start) / 2
        old_start, old_end = new_start, wall.end
    else:
        raise ValueError(end)

    # align height to the existing wall base
    segment_center = segment_center.copy()
    segment_center[2] = wall.base_z + wall.height / 2

    new_points = _box_points(
        segment_center, length=abs(delta_length),
        height=wall.height, thickness=wall.thickness,
        dir_vec=d, normal=wall.plane_normal, density_m=density_m)

    # update wall
    new_wall = WallObject(
        id=wall.id, start=old_start, end=old_end, direction=d,
        length=float(np.linalg.norm(old_end - old_start)),
        height=wall.height, thickness=wall.thickness,
        thickness_estimated=wall.thickness_estimated,
        base_z=wall.base_z, plane_normal=wall.plane_normal,
        confidence=wall.confidence, source_segment_id=wall.source_segment_id,
        meta={**wall.meta, "extended_by": wall.meta.get("extended_by", 0.0) + delta_length})

    # merge cloud
    merged = np.concatenate([scene.points, new_points], axis=0)
    synth = np.concatenate([scene.synthetic_mask, np.ones(len(new_points), dtype=bool)])
    colors = None
    if scene.colors is not None:
        pad = np.tile(np.array([[0.6, 0.6, 0.6]]), (len(new_points), 1))
        colors = np.concatenate([scene.colors, pad], axis=0)

    walls = list(scene.walls); walls[idx] = new_wall
    return Scene(points=merged, colors=colors, segments=scene.segments,
                 walls=walls, synthetic_mask=synth)


def replace_terrain(scene: Scene, polygon_xy: np.ndarray, new_z: float,
                    density_m: float = 0.05) -> Scene:
    """Replace ground points inside `polygon_xy` (Mx2) with a flat patch at `new_z`."""
    from matplotlib.path import Path as MplPath
    poly = MplPath(polygon_xy)
    keep = ~poly.contains_points(scene.points[:, :2])
    # collect ground-semantic segments to know which to remove
    ground_seg_pts = np.zeros(len(scene.points), dtype=bool)
    for s in scene.segments:
        if s.semantic in ("ground", "road"):
            ground_seg_pts[s.indices] = True
    remove = (~keep) & ground_seg_pts
    kept_idx = np.where(~remove)[0]
    pts = scene.points[kept_idx]
    cols = scene.colors[kept_idx] if scene.colors is not None else None
    synth = scene.synthetic_mask[kept_idx]

    # generate new patch: regular grid inside polygon
    xmin, ymin = polygon_xy.min(axis=0); xmax, ymax = polygon_xy.max(axis=0)
    xs = np.arange(xmin, xmax, density_m); ys = np.arange(ymin, ymax, density_m)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = poly.contains_points(grid)
    new_pts = np.zeros((inside.sum(), 3))
    new_pts[:, 0:2] = grid[inside]; new_pts[:, 2] = new_z

    merged = np.concatenate([pts, new_pts], axis=0)
    synth_all = np.concatenate([synth, np.ones(len(new_pts), dtype=bool)])
    colors_all = None
    if cols is not None:
        pad = np.tile(np.array([[0.4, 0.35, 0.25]]), (len(new_pts), 1))
        colors_all = np.concatenate([cols, pad], axis=0)

    # segments get invalidated; caller can re-segment or keep wall-only segments.
    return Scene(points=merged, colors=colors_all, segments=[],
                 walls=scene.walls, synthetic_mask=synth_all)
