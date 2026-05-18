"""Sanity-check a dataset loader: load N random tiles, visualize them colored
by our 6-class unified taxonomy. Catch mapping bugs BEFORE you train.

Run:
    python -m okolis_ai.scripts.verify_dataset \
        --config okolis_ai/configs/randlanet.yaml \
        --dataset toronto3d --num 3

Prints per-class point counts per tile, then opens an Open3D window per tile.
Close the window to see the next tile.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import yaml

from ..datasets.builders import _REGISTRY
from ..fusion.hybrid import CLASSES
from ..interaction.viewer import CLASS_COLORS

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("Install open3d to visualize.")


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    cols = np.zeros((len(labels), 3), dtype=np.float32)
    for i, name in enumerate(CLASSES):
        cols[labels == i] = CLASS_COLORS[name]
    return cols


def show_tile(xyz: np.ndarray, labels: np.ndarray, title: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_labels(labels))
    # Print distribution
    counts = np.bincount(labels, minlength=len(CLASSES))
    total = counts.sum()
    print(f"\n=== {title} ({total} pts) ===")
    for i, name in enumerate(CLASSES):
        pct = 100.0 * counts[i] / max(total, 1)
        print(f"  {i} {name:10s}  {counts[i]:>9d}   {pct:5.1f}%")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=list(_REGISTRY.keys()))
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    conf = cfg["datasets"][args.dataset]
    cls, splitter = _REGISTRY[args.dataset]
    split = splitter(conf["root"])
    files = split[args.split]
    if not files:
        raise SystemExit(f"No files for split={args.split}")

    kwargs = dict(crop_points=cfg.get("crop_points", 65536),
                  voxel=cfg.get("voxel", 0.03),
                  augment=False, modality_dropout=False)
    if args.dataset == "semantickitti":
        kwargs["raw_to_learning"] = split.get("raw_to_learning", {})
    if args.dataset == "botanicgarden":
        kwargs["label_field"] = conf.get("label_field", "scalar_class")

    ds = cls(files, **kwargs)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), size=min(args.num, len(ds)), replace=False)
    for i in picks:
        xyz, feats, labels = ds[int(i)]
        show_tile(xyz, labels, f"{args.dataset}[{i}] = {files[int(i)].name}")


if __name__ == "__main__":
    main()
