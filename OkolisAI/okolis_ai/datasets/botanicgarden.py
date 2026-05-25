"""BotanicGarden loader.

The official BotanicGarden release ships semantic clouds as PLY files with a
per-point `scalar_class` (or `label`) field. Format varies by release (some
tarballs use .pcd). This loader handles both .ply and .pcd defensively.

We assume the user has extracted `semantic_clouds/` with .ply files. If your
release uses different field names, set `label_field=...` in the config.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from .merged import PointCloudTileDataset, LoadedScan
from .label_maps import BOTANIC_MAP, apply_map


def _read_ply(path: Path, label_field: str) -> LoadedScan:
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    # label field search
    for key in (label_field, "scalar_class", "class", "label", "scalar_Label"):
        if key in v.dtype.names:
            raw = np.asarray(v[key], dtype=np.int64)
            break
    else:
        raise ValueError(f"No label field in {path}. Fields={v.dtype.names}")
    labels = apply_map(raw, BOTANIC_MAP)
    return LoadedScan(xyz=xyz, rgb=rgb, intensity=None, labels=labels)


def _read_pcd(path: Path) -> LoadedScan:
    # Minimal ASCII/binary PCD reader (header parse + numpy).
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.startswith(b"DATA"): break
        body = f.read()
    ht = header.decode(errors="ignore")
    fields = next(l for l in ht.splitlines() if l.startswith("FIELDS")).split()[1:]
    sizes  = [int(x) for x in next(l for l in ht.splitlines() if l.startswith("SIZE")).split()[1:]]
    types  = next(l for l in ht.splitlines() if l.startswith("TYPE")).split()[1:]
    counts = [int(x) for x in next(l for l in ht.splitlines() if l.startswith("COUNT")).split()[1:]]
    n      = int(next(l for l in ht.splitlines() if l.startswith("POINTS")).split()[1])
    data_line = next(l for l in ht.splitlines() if l.startswith("DATA")).split()[1]
    if data_line != "binary":
        raise NotImplementedError("Only binary PCD supported; re-export.")
    dtype = np.dtype([(fn, {"F": "f", "U": "u", "I": "i"}[t] + str(s))
                      for fn, t, s, c in zip(fields, types, sizes, counts)])
    arr = np.frombuffer(body, dtype=dtype, count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(k in arr.dtype.names for k in ("r", "g", "b")):
        rgb = np.stack([arr["r"], arr["g"], arr["b"]], axis=1).astype(np.float32) / 255.0
    label_name = next((k for k in ("label", "class", "semantic") if k in arr.dtype.names), None)
    if label_name is None:
        raise ValueError(f"No label field in PCD: {arr.dtype.names}")
    labels = apply_map(arr[label_name].astype(np.int64), BOTANIC_MAP)
    return LoadedScan(xyz=xyz, rgb=rgb, intensity=None, labels=labels)


class BotanicGardenDataset(PointCloudTileDataset):
    def __init__(self, *args, label_field: str = "scalar_class", **kw):
        super().__init__(*args, **kw)
        self.label_field = label_field

    def _load_scan(self, path: Path) -> LoadedScan:
        if path.suffix.lower() == ".pcd":
            return _read_pcd(path)
        return _read_ply(path, self.label_field)


def default_split(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    files = sorted(list(root.rglob("*.ply")) + list(root.rglob("*.pcd")))
    if not files:
        raise FileNotFoundError(f"No .ply/.pcd under {root}")
    # BotanicGarden has no published split; scene-level 80/10/10.
    rng = np.random.default_rng(0)
    idx = np.arange(len(files)); rng.shuffle(idx)
    n = len(files)
    ntr = int(0.8 * n); nva = int(0.1 * n)
    tr = [files[i] for i in idx[:ntr]]
    va = [files[i] for i in idx[ntr:ntr+nva]]
    te = [files[i] for i in idx[ntr+nva:]]
    return {"train": tr, "val": va, "test": te}
