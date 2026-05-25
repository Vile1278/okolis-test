"""Custom iPhone scans: .ply files with scalar label field.

Export from CloudCompare: save as binary PLY with the label scalar named
`label` or `scalar_Label`. Labels must ALREADY be in the unified 0..5 space
(no remap). Use the CloudCompare "Scalar field" editor to assign:
    1=ground, 2=road, 3=wall, 4=nature, 5=object, 0=unlabeled
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .merged import PointCloudTileDataset, LoadedScan


def _read_custom_ply(path: Path) -> LoadedScan:
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    for key in ("label", "scalar_Label", "class"):
        if key in v.dtype.names:
            labels = np.asarray(v[key], dtype=np.int64)
            break
    else:
        raise ValueError(f"No label field in {path}")
    # clip to valid range; anything else → unlabeled
    labels = np.where((labels >= 0) & (labels <= 5), labels, 0)
    return LoadedScan(xyz=xyz, rgb=rgb, intensity=None, labels=labels)


class CustomIPhoneDataset(PointCloudTileDataset):
    def _load_scan(self, path: Path) -> LoadedScan:
        return _read_custom_ply(path)


def default_split(root: str | Path, seed: int = 0) -> dict[str, list[Path]]:
    root = Path(root)
    files = sorted(root.rglob("*.ply"))
    if not files:
        return {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files)); rng.shuffle(idx)
    n = len(files)
    ntr = int(0.6 * n); nva = int(0.2 * n)
    tr = [files[i] for i in idx[:ntr]]
    va = [files[i] for i in idx[ntr:ntr+nva]]
    te = [files[i] for i in idx[ntr+nva:]]
    return {"train": tr, "val": va, "test": te}
