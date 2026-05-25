"""Shared helpers: feature unification, per-tile ground height, augmentation."""
from __future__ import annotations
import numpy as np

# Unified feature layout (must match configs/randlanet.yaml in_feat_dim).
#   0: R           (0..1, 0 if unknown)
#   1: G
#   2: B
#   3: intensity   (0..1, 0 if unknown)
#   4: height-above-ground (meters)
FEAT_DIM = 5


def pack_features(rgb: np.ndarray | None,
                  intensity: np.ndarray | None,
                  height_above_ground: np.ndarray) -> np.ndarray:
    n = len(height_above_ground)
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    if rgb is not None:
        feats[:, 0:3] = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    if intensity is not None:
        feats[:, 3] = np.clip(intensity.astype(np.float32), 0.0, 1.0)
    feats[:, 4] = height_above_ground.astype(np.float32)
    return feats


def height_above_ground_from_labels(xyz: np.ndarray, labels: np.ndarray,
                                    ground_label: int = 1,
                                    cell: float = 1.0) -> np.ndarray:
    """Per-cell (XY grid) median-Z of ground points; height = z - local_ground_z.
    Cheap and robust. If no ground points in a cell, fall back to global ground
    z or per-tile 5th percentile of z."""
    mask = labels == ground_label
    if not mask.any():
        return (xyz[:, 2] - np.percentile(xyz[:, 2], 5)).astype(np.float32)
    g = xyz[mask]
    ix = np.floor((xyz[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    max_iy = int(iy.max()) + 1
    keys = ix * max_iy + iy

    gx = np.floor((g[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    gy = np.floor((g[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    gk = gx * max_iy + gy

    # cell → median z
    order = np.argsort(gk)
    sk = gk[order]; sz = g[order, 2]
    uniq, starts = np.unique(sk, return_index=True)
    ends = np.r_[starts[1:], len(sk)]
    table = {int(k): float(np.median(sz[s:e])) for k, s, e in zip(uniq, starts, ends)}
    fallback = float(np.median(g[:, 2]))
    floor = np.array([table.get(int(k), fallback) for k in keys], dtype=np.float32)
    return (xyz[:, 2] - floor).astype(np.float32)


def modality_dropout(feats: np.ndarray,
                     drop_rgb_p: float = 0.3,
                     drop_intensity_p: float = 0.3,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly zero entire RGB or intensity channels per tile.
    Teaches the model to handle domain gaps (iPhone has no intensity; some
    scans have no RGB). Height is never dropped — always a reliable feature."""
    rng = rng or np.random.default_rng()
    feats = feats.copy()
    if rng.random() < drop_rgb_p:
        feats[:, 0:3] = 0.0
    if rng.random() < drop_intensity_p:
        feats[:, 3] = 0.0
    return feats
