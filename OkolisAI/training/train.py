"""Training entrypoint for RandLA-Net on the unified dataset concat.

Run:
    python -m okolis_ai.training.train --config okolis_ai/configs/randlanet.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    raise SystemExit("Install torch to train.")

from ..ml.randlanet.model import RandLANet
from ..datasets.builders import build_datasets, build_sampler
from .losses import LovaszSoftmax, class_weighted_ce

def _collate(batch):
    import numpy as np
    def _t(x):
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    xyz = torch.stack([_t(b[0]).float() for b in batch])
    feats = torch.stack([_t(b[1]).float() for b in batch])
    labels = torch.stack([_t(b[2]).long() for b in batch])
    feats = torch.nan_to_num(feats, nan=0.0)
    return xyz, feats, labels

def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    train_ds, val_ds, per_sample_w = build_datasets(cfg)
    sampler = build_sampler(per_sample_w, num_samples=cfg.get("steps_per_epoch",
                                                              len(per_sample_w)))
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=cfg.get("num_workers", 4), collate_fn=_collate, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, collate_fn=_collate) if val_ds else None

    model = RandLANet(in_feat_dim=cfg.get("in_feat_dim", 5),
                      num_classes=cfg["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    lovasz = LovaszSoftmax(ignore_index=0)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    best_miou = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0; nb = 0
        for xyz, feats, labels in train_loader:
            xyz = xyz.to(device); feats = feats.to(device); labels = labels.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xyz, feats)                 # (B,N,C)
                ce = class_weighted_ce(logits, labels,
                                       weights=cfg.get("class_weights"))
                lv = lovasz(logits, labels)
                loss = ce + 0.5 * lv
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += float(loss.item()); nb += 1
        sched.step()

        miou = evaluate(model, val_loader, device, cfg["num_classes"]) if val_loader else 0.0
        print(f"epoch {epoch:3d}  loss={total/max(nb,1):.3f}  mIoU={miou:.3f}  lr={opt.param_groups[0]['lr']:.5f}")
        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou,
                        "cfg": cfg}, out / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, out / "last.pt")
    print(f"[train] best mIoU: {best_miou:.3f}")


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    inter = torch.zeros(num_classes); union = torch.zeros(num_classes)
    for xyz, feats, labels in loader:
        xyz = xyz.to(device); feats = feats.to(device); labels = labels.to(device)
        logits = model(xyz, feats)
        pred = logits.argmax(dim=-1)
        for c in range(1, num_classes):       # skip "unlabeled"
            p = (pred == c); t = (labels == c)
            inter[c] += (p & t).sum().cpu()
            union[c] += (p | t).sum().cpu()
    iou = inter[1:] / union[1:].clamp(min=1)
    print("  per-class IoU:", [f"{v:.2f}" for v in iou.tolist()])
    return float(iou.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    train(cfg)


if __name__ == "__main__":
    main()
