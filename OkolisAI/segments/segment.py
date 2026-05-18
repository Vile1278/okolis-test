"""Unified Segment dataclass — the atomic unit of the Scene."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import uuid
import numpy as np

from ..geometry.features import SegmentFeatures

SemanticLabel = Literal[
    "unlabeled", "ground", "road", "sidewalk",
    "building", "fence", "vegetation", "vehicle"
]
SegmentKind   = Literal["plane", "cluster", "ground"]


@dataclass
class Segment:
    id: str
    kind: SegmentKind
    indices: np.ndarray
    features: SegmentFeatures
    semantic: SemanticLabel = "unlabeled"
    confidence: float = 0.0
    normal: Optional[np.ndarray] = None      # for planes
    plane: Optional[np.ndarray] = None       # [a,b,c,d]
    meta: dict = field(default_factory=dict)

    @staticmethod
    def new(kind: SegmentKind, indices: np.ndarray, features: SegmentFeatures,
            **kw) -> "Segment":
        return Segment(id=f"seg_{uuid.uuid4().hex[:10]}",
                       kind=kind, indices=indices, features=features, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["indices"] = self.indices.tolist()
        if self.normal is not None:
            d["normal"] = self.normal.tolist()
        if self.plane is not None:
            d["plane"] = self.plane.tolist()
        # features contains numpy arrays
        f = d["features"]
        for k, v in list(f.items()):
            if isinstance(v, np.ndarray):
                f[k] = v.tolist()
        return d
