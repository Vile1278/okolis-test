"""Segment → WallObject. Robust endpoint/length/thickness estimation."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import uuid
import numpy as np


@dataclass
class WallObject:
    id: str
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    height: float
    thickness: float
    thickness_estimated: bool
    base_z: float
    plane_normal: np.ndarray
    confidence: float
    source_segment_id: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "direction", "plane_normal"):
            d[k] = np.asarray(getattr(self, k)).tolist()
        return d


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    world_up = np.array([0.0, 0.0, 1.0])
    # u = in-plane horizontal axis
    u = np.cross(world_up, n)
    if np.linalg.norm(u) < 1e-6:
        # plane is horizontal — not a wall, but still return a valid basis
        u = np.array([1.0, 0.0, 0.0]) - n * n[0]
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v, n


def extract_wall(points: np.ndarray,
                 plane_normal: np.ndarray,
                 source_segment_id: str,
                 confidence: float = 1.0,
                 all_planes: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
                 default_thickness: float = 0.15) -> WallObject:
    """Build a WallObject from its points and plane normal.

    `all_planes`: optional list of (normal, centroid) for neighbouring planes,
    used to estimate thickness by detecting a parallel twin plane."""
    u, v, n = _plane_basis(plane_normal)
    centroid = points.mean(axis=0)
    local = points - centroid
    uu = local @ u          # along wall
    vv = local @ v          # vertical

    # Robust extents (1st/99th percentile — rejects stragglers)
    u_lo, u_hi = np.percentile(uu, [1, 99])
    v_lo, v_hi = np.percentile(vv, [1, 99])
    length = float(u_hi - u_lo)
    height = float(v_hi - v_lo)

    # PCA refinement on (uu, vv) to correct tilted walls: principal axis should
    # align with u; if not, rotate within the plane.
    M = np.stack([uu, vv], axis=1)
    cov = np.cov(M.T) + np.eye(2) * 1e-9
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    principal_2d = vec[:, order[0]]        # in (u,v) space
    # world-frame principal direction
    direction = principal_2d[0] * u + principal_2d[1] * v
    # project along refined direction for final length
    proj = local @ direction
    p_lo, p_hi = np.percentile(proj, [1, 99])
    length = float(p_hi - p_lo)
    start = centroid + direction * p_lo
    end = centroid + direction * p_hi

    # Thickness: look for parallel twin plane within 0.05–0.5 m
    thickness = default_thickness
    thickness_estimated = True
    if all_planes:
        best = None
        for other_n, other_c in all_planes:
            other_n = other_n / (np.linalg.norm(other_n) + 1e-12)
            if abs(other_n @ n) < 0.95:
                continue
            d = abs((other_c - centroid) @ n)
            if 0.05 <= d <= 0.5:
                if best is None or d < best:
                    best = d
        if best is not None:
            thickness = float(best)
            thickness_estimated = False

    # horizontalize the direction vector for downstream editing
    dir_h = direction.copy()
    dir_h[2] = 0.0
    if np.linalg.norm(dir_h) > 1e-6:
        dir_h /= np.linalg.norm(dir_h)
    else:
        dir_h = direction / (np.linalg.norm(direction) + 1e-12)

    return WallObject(
        id=f"wall_{uuid.uuid4().hex[:10]}",
        start=start, end=end, direction=dir_h,
        length=length, height=height, thickness=thickness,
        thickness_estimated=thickness_estimated,
        base_z=float(points[:, 2].min()),
        plane_normal=n, confidence=confidence,
        source_segment_id=source_segment_id)
