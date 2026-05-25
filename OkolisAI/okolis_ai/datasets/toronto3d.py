"""Toronto-3D loader.

Format: 4 tiles L001.ply .. L004.ply. Each is a binary PLY with custom scalars:
  x, y, z, red, green, blue, scalar_Intensity, scalar_Label
Standard split (per Toronto-3D paper): train = {L001, L003, L004}, test = L002.

We further carve val from train. Labels remap via TORONTO3D_MAP.

Dependencies: `plyfile` (pip install plyfile). Open3D discards custom scalars,
so plyfile is the correct reader here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .merged import PointCloudTileDataset, LoadedScan
from .label_maps import TORONTO3D_MAP, apply_map


def _read_toronto_ply(path: Path) -> LoadedScan:
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    intensity = None
    for key in ("scalar_Intensity", "intensity", "Intensity"):
        if key in v.dtype.names:
            raw = np.asarray(v[key], dtype=np.float32)
            # Toronto intensity is uint16-ish; normalize to [0,1]
            mx = max(float(raw.max()), 1.0)
            intensity = (raw / mx).astype(np.float32)
            break
    label_raw = None
    for key in ("scalar_Label", "label", "class"):
        if key in v.dtype.names:
            label_raw = np.asarray(v[key], dtype=np.int64)
            break
    if label_raw is None:
        raise ValueError(f"No label field found in {path}")
    labels = apply_map(label_raw, TORONTO3D_MAP)
    return LoadedScan(xyz=xyz, rgb=rgb, intensity=intensity, labels=labels)


class Toronto3DDataset(PointCloudTileDataset):
    """Each `scan_path` is one of the big L00x.ply tiles. Since a single tile
    is huge (10M+ pts) we crop randomly per __getitem__ call; increase
    `samples_per_scan` in the builder to sample many crops per epoch."""

    def _load_scan(self, path: Path) -> LoadedScan:
        return _read_toronto_ply(path)


def default_split(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    tiles = sorted(root.glob("L00*.ply"))
    if not tiles:
        raise FileNotFoundError(f"No L00*.ply tiles under {root}")
    by_name = {p.stem: p for p in tiles}
    train = [by_name[k] for k in ("L001", "L003", "L004") if k in by_name]
    test  = [by_name[k] for k in ("L002",) if k in by_name]
    # carve val from train by taking 10% random crops via upstream oversampling;
    # cleanest is to hold out one file — but Toronto has only 3 train files.
    # Solution: use L004 as val.
    val   = [by_name.get("L004")] if "L004" in by_name else []
    train = [p for p in train if p.stem != "L004"]
    return {"train": train, "val": val, "test": test}
