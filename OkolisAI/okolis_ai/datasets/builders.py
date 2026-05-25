"""Assemble a unified training/val DataLoader from the yaml config."""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    from torch.utils.data import ConcatDataset, WeightedRandomSampler
except ImportError:
    ConcatDataset = object  # type: ignore
    WeightedRandomSampler = None  # type: ignore

from . import toronto3d, semantickitti, botanicgarden, custom_iphone


_REGISTRY = {
    "toronto3d":    (toronto3d.Toronto3DDataset,    toronto3d.default_split),
    "semantickitti":(semantickitti.SemanticKITTIDataset, semantickitti.default_split),
    "botanicgarden":(botanicgarden.BotanicGardenDataset, botanicgarden.default_split),
    "custom":       (custom_iphone.CustomIPhoneDataset,  custom_iphone.default_split),
}


def build_datasets(cfg: dict) -> tuple["ConcatDataset", "ConcatDataset", list[float]]:
    """Returns (train_concat, val_concat, per_sample_weights_for_train).
    `per_sample_weights_for_train` feeds WeightedRandomSampler so each dataset
    contributes according to its configured weight regardless of file count."""
    train_parts, val_parts, weights = [], [], []
    ds_sizes = []

    for name, conf in cfg["datasets"].items():
        if name not in _REGISTRY:
            print(f"[builders] unknown dataset '{name}', skipping")
            continue
        cls, splitter = _REGISTRY[name]
        root = conf.get("root")
        if root is None:
            continue
        try:
            split = splitter(root)
        except FileNotFoundError as e:
            print(f"[builders] {name}: {e} — skipping")
            continue
        if not split["train"]:
            print(f"[builders] {name}: no train files found — skipping")
            continue

        kwargs = dict(crop_points=cfg.get("crop_points", 65536),
                      voxel=cfg.get("voxel", 0.03),
                      augment=cfg.get("augment", True),
                      modality_dropout=cfg.get("modality_dropout", True))
        # dataset-specific extras
        if name == "semantickitti":
            kwargs["raw_to_learning"] = split.get("raw_to_learning", {})
        if name == "botanicgarden":
            kwargs["label_field"] = conf.get("label_field", "scalar_class")

        tr = cls(split["train"], **kwargs)
        va = cls(split["val"], **{**kwargs, "augment": False, "modality_dropout": False})
        train_parts.append(tr)
        if len(va) > 0:
            val_parts.append(va)
        w = float(conf.get("weight", 1.0))
        weights.extend([w] * len(tr))
        ds_sizes.append((name, len(tr), len(va) if split["val"] else 0, w))
        print(f"[builders] {name}: train={len(tr)} val={len(va)} weight={w}")

    if not train_parts:
        raise RuntimeError("No datasets could be loaded. Check `datasets.*.root` paths.")

    train = ConcatDataset(train_parts)
    val = ConcatDataset(val_parts) if val_parts else None

    # Normalize weights and return as per-sample vector for WeightedRandomSampler.
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum() * len(weights)
    return train, val, weights.tolist()


def build_sampler(weights: list[float], num_samples: int | None = None):
    if WeightedRandomSampler is None:
        raise ImportError("Install torch.")
    n = num_samples or len(weights)
    return WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)
