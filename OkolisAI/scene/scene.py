"""Scene container: segments, walls, spatial index for picking."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import numpy as np
import open3d as o3d

from ..segments.segment import Segment
from ..walls.wall_extractor import WallObject


@dataclass
class Scene:
    points: np.ndarray                         # (N,3) working cloud (downsampled)
    colors: Optional[np.ndarray] = None
    segments: list[Segment] = field(default_factory=list)
    walls: list[WallObject] = field(default_factory=list)
    synthetic_mask: np.ndarray = None          # set in __post_init__
    _kd: o3d.geometry.KDTreeFlann | None = None

    def __post_init__(self):
        if self.synthetic_mask is None:
            self.synthetic_mask = np.zeros(len(self.points), dtype=bool)
        self._rebuild_index()

    # ---------- spatial index ----------
    def _rebuild_index(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        self._kd = o3d.geometry.KDTreeFlann(pcd)
        # point → segment lookup
        self._pt2seg = -np.ones(len(self.points), dtype=np.int64)
        for i, seg in enumerate(self.segments):
            self._pt2seg[seg.indices] = i

    def point_to_segment(self, i: int) -> Segment | None:
        j = int(self._pt2seg[i])
        return self.segments[j] if j >= 0 else None

    def nearest_point(self, p: np.ndarray, k: int = 1):
        _, idx, _ = self._kd.search_knn_vector_3d(p.astype(np.float64), k)
        return int(idx[0])

    # ---------- mutation ----------
    def replace(self, **kw) -> "Scene":
        """Shallow-copy-with-updates (functional edits return new scenes)."""
        new = Scene(
            points=self.points, colors=self.colors,
            segments=list(self.segments), walls=list(self.walls),
            synthetic_mask=self.synthetic_mask.copy())
        for k, v in kw.items():
            setattr(new, k, v)
        new._rebuild_index()
        return new

    # ---------- persistence ----------
    def save(self, path: str | Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "cloud.npz",
                            points=self.points,
                            colors=self.colors if self.colors is not None else np.zeros((0, 3)),
                            synthetic=self.synthetic_mask)
        with open(path / "scene.json", "w", encoding="utf-8") as f:
            json.dump({
                "segments": [s.to_dict() for s in self.segments],
                "walls":    [w.to_dict() for w in self.walls],
            }, f, indent=2)
