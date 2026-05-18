"""Ray picking: viewport click → scene point → segment → object."""
from __future__ import annotations
import numpy as np

from ..scene.scene import Scene


def pick_segment(scene: Scene, world_point: np.ndarray) -> str | None:
    """Given a world-space point (e.g. from a picked nearest-neighbor in the viewer),
    return the enclosing segment id."""
    i = scene.nearest_point(world_point)
    seg = scene.point_to_segment(i)
    return seg.id if seg else None


def pick_wall(scene: Scene, world_point: np.ndarray) -> str | None:
    seg_id = pick_segment(scene, world_point)
    if seg_id is None:
        return None
    for w in scene.walls:
        if w.source_segment_id == seg_id:
            return w.id
    return None


def ray_to_world_point(ray_origin: np.ndarray, ray_direction: np.ndarray,
                       scene: Scene, max_dist: float = 100.0) -> np.ndarray | None:
    """Approximate ray-cloud intersection: step along the ray, find the nearest
    point, accept if it's within a tolerance of the ray. This is the robust
    and cheap approach that works for point clouds (no mesh needed)."""
    d = ray_direction / (np.linalg.norm(ray_direction) + 1e-12)
    best = None; best_t = 0.0; best_r = np.inf
    # step-march with KD-tree nearest query
    for t in np.linspace(0.1, max_dist, 200):
        p = ray_origin + d * t
        i = scene.nearest_point(p)
        q = scene.points[i]
        r = np.linalg.norm(q - p)
        if r < best_r:
            best_r = r; best = q; best_t = t
            if r < 0.05:
                return q
    # Accept only if we got reasonably close
    return best if best_r < 0.3 else None
