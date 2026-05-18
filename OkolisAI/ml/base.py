"""Segmenter interface: any model plugs in behind this."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Segmenter(ABC):
    num_classes: int

    @abstractmethod
    def predict(self, xyz: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        """Return per-point softmax (N, C)."""


class UniformSegmenter(Segmenter):
    """Debug/no-op segmenter: uniform probabilities. Lets the geometry-only
    pipeline run end-to-end before ML weights exist."""
    num_classes = 8

    def predict(self, xyz: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        n = len(xyz)
        return np.full((n, self.num_classes), 1.0 / self.num_classes, dtype=np.float32)
