"""Abstract dataset base + shared crop/augment logic.

Each concrete dataset subclasses `PointCloudTileDataset` and implements
`_load_scan(path) -> (xyz, rgb, intensity, labels_unified)` where labels are
ALREADY remapped to the 6-class unified taxonomy.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .common import pack_features, height_above_ground_from_labels, modality_dropout

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object  # type: ignore


@dataclass
class LoadedScan:
    xyz: np.ndarray                     # (N, 3) float32
    rgb: Optional[np.ndarray]           # (N, 3) 0..1 or None
    intensity: Optional[np.ndarray]     # (N,) 0..1 or None
    labels: np.ndarray                  # (N,) int64, unified 0..5


class PointCloudTileDataset(Dataset):
    """Shared logic: random anchor-radius crop, rotation/jitter/scale/drop
    augmentation, modality dropout, feature packing."""

    ground_label = 1  # in unified taxonomy

    def __init__(self, scan_paths: list[Path], crop_points: int = 65536,
                 voxel: float = 0.03, augment: bool = True,
                 modality_dropout: bool = True, seed: int | None = None):
        self.paths = list(scan_paths)
        self.crop = crop_points
        self.voxel = voxel
        self.augment = augment
        self.do_mod_drop = modality_dropout
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.paths)

    # ------- subclass hook -------
    def _load_scan(self, path: Path) -> LoadedScan:
        raise NotImplementedError

    # ------- crop & augment -------
    def _voxel_downsample(self, scan: LoadedScan) -> LoadedScan:
        """Cheap voxel-averaging downsample (no Open3D dep, safe in workers)."""
        v = self.voxel
        if v <= 0: return scan
        keys = np.floor(scan.xyz / v).astype(np.int64)
        # hash per-voxel
        k1 = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
        order = np.argsort(k1)
        sk = k1[order]
        uniq, starts = np.unique(sk, return_index=True)
        # take first point per voxel (random-order argsort gives reasonable mix)
        pick = order[starts]
        return LoadedScan(
            xyz=scan.xyz[pick],
            rgb=scan.rgb[pick] if scan.rgb is not None else None,
            intensity=scan.intensity[pick] if scan.intensity is not None else None,
            labels=scan.labels[pick])

    def _anchor_crop(self, scan: LoadedScan) -> LoadedScan:
        n = len(scan.xyz)
        if n == 0:
            raise ValueError("empty scan")
        if n <= self.crop:
            pad = self.rng.integers(0, n, size=self.crop - n)
            idx = np.concatenate([np.arange(n), pad])
        else:
            anchor = scan.xyz[self.rng.integers(0, n)]
            d2 = ((scan.xyz - anchor) ** 2).sum(axis=1)
            # k-th smallest → crop ball
            idx = np.argpartition(d2, self.crop)[:self.crop]
        return LoadedScan(
            xyz=scan.xyz[idx],
            rgb=scan.rgb[idx] if scan.rgb is not None else None,
            intensity=scan.intensity[idx] if scan.intensity is not None else None,
            labels=scan.labels[idx])

    def _augment_geo(self, scan: LoadedScan) -> LoadedScan:
        xyz = scan.xyz.copy()
        # rotation around Z
        t = self.rng.uniform(-np.pi, np.pi)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
        # scale
        xyz *= self.rng.uniform(0.9, 1.1)
        # jitter
        xyz += self.rng.normal(0.0, 0.01, xyz.shape).astype(np.float32)
        return LoadedScan(xyz=xyz, rgb=scan.rgb, intensity=scan.intensity, labels=scan.labels)

    # ------- pipeline -------
    def __getitem__(self, i):
        scan = self._load_scan(self.paths[i])
        scan = self._voxel_downsample(scan)
        if self.augment:
            scan = self._augment_geo(scan)
        scan = self._anchor_crop(scan)

        # height-above-ground BEFORE we center XYZ
        h = height_above_ground_from_labels(
            scan.xyz, scan.labels, ground_label=self.ground_label, cell=1.0)

        # feature pack
        feats = pack_features(scan.rgb, scan.intensity, h)
        if self.do_mod_drop:
            feats = modality_dropout(feats, rng=self.rng)

        # center XYZ (after feature pack — height uses world z)
        xyz = scan.xyz - scan.xyz.mean(axis=0)
        return (xyz.astype(np.float32),
                feats.astype(np.float32),
                scan.labels.astype(np.int64))
