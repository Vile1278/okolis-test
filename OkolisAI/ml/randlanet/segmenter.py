"""RandLA-Net adapter for the Segmenter interface."""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:  # keep geometry pipeline usable without torch installed
    torch = None

from ..base import Segmenter


class RandLANetSegmenter(Segmenter):
    num_classes = 8

    def __init__(self, weights: str | Path | None = None, device: str = "cuda",
                 points_per_tile: int = 16384, in_feat_dim: int = 5):
        if torch is None:
            raise ImportError("Install torch to use RandLANetSegmenter")
        from .model import RandLANet
        self.device = device if torch.cuda.is_available() else "cpu"
        self.in_feat_dim = in_feat_dim

        # Load checkpoint and read config from it
        if weights is not None:
            checkpoint = torch.load(str(weights), map_location=self.device,
                                    weights_only=False)
            cfg = checkpoint.get("cfg", {})
            self.num_classes = cfg.get("num_classes", 8)
            actual_in_feat_dim = cfg.get("in_feat_dim", in_feat_dim)
        else:
            checkpoint = None
            actual_in_feat_dim = in_feat_dim

        # Extract PTv3 architecture config from checkpoint if available
        ptv3_cfg = cfg.get("ptv3", {})
        model_kwargs = dict(
            in_feat_dim=actual_in_feat_dim,
            num_classes=self.num_classes,
        )
        if ptv3_cfg:
            model_kwargs.update(
                dims=tuple(ptv3_cfg["dims"]),
                num_heads=tuple(ptv3_cfg["num_heads"]),
                depths=tuple(ptv3_cfg["depths"]),
                window_size=ptv3_cfg.get("window_size", 256),
                grid_sizes=tuple(ptv3_cfg["grid_sizes"]),
                serialize_grid=ptv3_cfg.get("serialize_grid", 0.04),
                drop=ptv3_cfg.get("drop", 0.0),
            )

        self.model = RandLANet(**model_kwargs).to(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint.get("model", checkpoint))
        self.model.eval()
        self.N = points_per_tile

    @torch.inference_mode() if torch is not None else (lambda f: f)
    def predict(self, xyz: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        n = len(xyz)
        if features is None:
            features = np.zeros((n, self.in_feat_dim), dtype=np.float32)
        feats = features.astype(np.float32)

        probs = np.zeros((n, self.num_classes), dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)

        rng = np.random.default_rng(0)
        cursor = 0
        while cursor < n:
            take = min(self.N, n - cursor)
            idx = np.arange(cursor, cursor + take)
            if take < self.N:
                pad = rng.choice(n, size=self.N - take, replace=True)
                idx_all = np.concatenate([idx, pad])
            else:
                idx_all = idx
            xyz_t = torch.from_numpy(xyz[idx_all]).float().unsqueeze(0).to(self.device)
            f_t = torch.from_numpy(feats[idx_all]).float().unsqueeze(0).to(self.device)
            f_t = torch.nan_to_num(f_t, nan=0.0)
            logits = self.model(xyz_t, f_t)                   # (1,N,C)
            p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            probs[idx] += p[:take]
            counts[idx] += 1
            cursor += take
        counts = np.maximum(counts, 1)
        return probs / counts[:, None]
