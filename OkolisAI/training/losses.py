"""Losses: class-weighted cross entropy + Lovász-Softmax (boundary-aware)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weighted_ce(logits: torch.Tensor, labels: torch.Tensor,
                      weights=None, ignore_index: int = 0) -> torch.Tensor:
    # logits: (B,N,C); labels: (B,N)
    B, N, C = logits.shape
    w = None
    if weights is not None:
        w = torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1),
                           weight=w, ignore_index=ignore_index)


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        B, N, C = probs.shape
        probs = probs.reshape(-1, C)
        labels = labels.reshape(-1)
        keep = labels != self.ignore
        if keep.sum() == 0:
            return logits.sum() * 0
        probs = probs[keep]; labels = labels[keep]
        losses = []
        for c in range(C):
            if c == self.ignore: continue
            fg = (labels == c).float()
            if fg.sum() == 0: continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            losses.append((errors_sorted * grad).sum())
        return torch.stack(losses).mean() if losses else logits.sum() * 0
