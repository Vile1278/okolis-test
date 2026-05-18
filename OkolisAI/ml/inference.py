"""Tiled inference for arbitrarily large outdoor clouds.

Strategy
--------
1. Split the cloud into overlapping XY tiles of ~65 k points each.
2. For each tile, run the Segmenter. Accumulate softmax into a per-point
   buffer (sum + count).
3. Average logits on overlaps → stable boundaries.
"""
from __future__ import annotations
import numpy as np
from .base import Segmenter


def _tile_indices(xyz: np.ndarray, tile_xy: float, overlap: float) -> list[np.ndarray]:
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    x1, y1 = xyz[:, 0].max(), xyz[:, 1].max()
    step = tile_xy * (1 - overlap)
    tiles: list[np.ndarray] = []
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            m = ((xyz[:, 0] >= x) & (xyz[:, 0] < x + tile_xy) &
                 (xyz[:, 1] >= y) & (xyz[:, 1] < y + tile_xy))
            if m.sum() > 200:
                tiles.append(np.where(m)[0])
            y += step
        x += step
    if not tiles:
        tiles = [np.arange(len(xyz))]
    return tiles


def segment_cloud(segmenter: Segmenter,
                  xyz: np.ndarray, features: np.ndarray | None = None,
                  tile_xy: float = 15.0, overlap: float = 0.1) -> np.ndarray:
    C = segmenter.num_classes
    n = len(xyz)
    acc = np.zeros((n, C), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.int32)
    for idx in _tile_indices(xyz, tile_xy, overlap):
        sub_xyz = xyz[idx]
        sub_f = features[idx] if features is not None else None
        p = segmenter.predict(sub_xyz, sub_f)
        acc[idx] += p
        cnt[idx] += 1
    cnt = np.maximum(cnt, 1)
    probs = acc / cnt[:, None]
    # renormalize (tiny drift from averaging)
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-9)
    return probs
