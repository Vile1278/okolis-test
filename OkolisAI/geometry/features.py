"""Per-segment geometric descriptors used by the fusion layer."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass(eq=False)
class SegmentFeatures:
    n_points: int
    centroid: np.ndarray
    extent: np.ndarray         # (3,) bbox size
    verticality: float         # 0..1
    planarity: float           # 0..1  from PCA eigenvalues
    linearity: float
    sphericity: float
    height_range: float        # max_z - min_z
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    obb_axes: np.ndarray = field(default_factory=lambda: np.eye(3))


def compute(points: np.ndarray) -> SegmentFeatures:
    if len(points) < 3:
        c = points.mean(axis=0) if len(points) else np.zeros(3)
        return SegmentFeatures(
            n_points=len(points), centroid=c, extent=np.zeros(3),
            verticality=0.0, planarity=0.0, linearity=0.0, sphericity=0.0,
            height_range=0.0, aabb_min=c, aabb_max=c)
    c = points.mean(axis=0)
    X = points - c
    cov = (X.T @ X) / len(points)
    eig, vec = np.linalg.eigh(cov)
    eig = np.maximum(eig, 1e-12)
    order = np.argsort(eig)[::-1]  # l1 >= l2 >= l3
    l1, l2, l3 = eig[order]
    axes = vec[:, order]
    linearity   = float((l1 - l2) / l1)
    planarity   = float((l2 - l3) / l1)
    sphericity  = float(l3 / l1)
    # Verticality: smallest eigenvector (surface normal) vs world up.
    normal = axes[:, 2]
    verticality = float(1.0 - abs(normal[2]))
    aabb_min = points.min(axis=0); aabb_max = points.max(axis=0)
    return SegmentFeatures(
        n_points=len(points), centroid=c, extent=(aabb_max - aabb_min),
        verticality=verticality, planarity=planarity, linearity=linearity,
        sphericity=sphericity,
        height_range=float(aabb_max[2] - aabb_min[2]),
        aabb_min=aabb_min, aabb_max=aabb_max, obb_axes=axes)
