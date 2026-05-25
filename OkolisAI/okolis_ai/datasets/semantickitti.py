"""SemanticKITTI loader.

Folder layout (standard):
    root/
      dataset/sequences/{00..21}/
        velodyne/*.bin     # float32  (x, y, z, intensity)
        labels/*.label     # uint32, low 16 bits = semantic id
    semantic-kitti.yaml    # learning_map: raw_id -> 0..19

We remap:  raw_id --learning_map--> 0..19 (their "learning" space)
           then --SEMKITTI_MAP (we only cover classes we care about)--> 0..5

Splits (per KITTI benchmark):
    train: 00 01 02 03 04 05 06 07 09 10
    val  : 08
    test : 11..21 (no labels public)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from .merged import PointCloudTileDataset, LoadedScan
from .label_maps import SEMKITTI_MAP, apply_map


TRAIN_SEQ = ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10")
VAL_SEQ   = ("08",)


def _load_kitti_yaml(root: Path) -> dict[int, int]:
    """Return `raw_id -> learning_id` from semantic-kitti.yaml."""
    import yaml
    candidates = [root / "semantic-kitti.yaml",
                  root.parent / "semantic-kitti.yaml",
                  Path(__file__).parent / "semantic-kitti.yaml"]
    for c in candidates:
        if c.exists():
            cfg = yaml.safe_load(open(c))
            return {int(k): int(v) for k, v in cfg["learning_map"].items()}
    # Hard-coded minimal map as last resort — enough for our 6-class task.
    # Source: KITTI learning_map collapsed.
    hardcoded = {
        0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6,
        31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0,
        60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0,
        252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}
    return hardcoded


def _read_kitti_scan(bin_path: Path, label_path: Path,
                     raw_to_learning: dict[int, int]) -> LoadedScan:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    intensity = pts[:, 3]
    # KITTI intensity is roughly 0..1 already; clip to be safe.
    intensity = np.clip(intensity, 0.0, 1.0)

    raw = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    # raw -> learning (0..19) -> unified (0..5)
    learning = np.zeros_like(raw, dtype=np.int64)
    for k, v in raw_to_learning.items():
        learning[raw == k] = v
    labels = apply_map(learning, SEMKITTI_MAP)

    return LoadedScan(xyz=xyz.astype(np.float32),
                      rgb=None, intensity=intensity, labels=labels)


class SemanticKITTIDataset(PointCloudTileDataset):
    """Each scan is one .bin+.label pair. KITTI sweeps are ~120k points →
    a single scan already roughly fits one training crop."""

    def __init__(self, *args, raw_to_learning: dict[int, int] | None = None, **kw):
        super().__init__(*args, **kw)
        self.raw_to_learning = raw_to_learning or {}

    def _load_scan(self, path: Path) -> LoadedScan:
        # We stored path to .bin; find matching .label
        bin_path = path
        label_path = Path(str(path).replace("velodyne", "labels").replace(".bin", ".label"))
        return _read_kitti_scan(bin_path, label_path, self.raw_to_learning)


def default_split(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    seq_root = root / "dataset" / "sequences"
    if not seq_root.exists():
        seq_root = root / "sequences"
    if not seq_root.exists():
        raise FileNotFoundError(f"Missing sequences/ under {root}")
    train, val = [], []
    for s in TRAIN_SEQ:
        train.extend(sorted((seq_root / s / "velodyne").glob("*.bin")))
    for s in VAL_SEQ:
        val.extend(sorted((seq_root / s / "velodyne").glob("*.bin")))
    return {"train": train, "val": val, "test": [],
            "raw_to_learning": _load_kitti_yaml(root)}
