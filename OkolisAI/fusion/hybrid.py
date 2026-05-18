"""Hybrid geometry + ML label assignment at the segment level.

Geometry priors tuned for iPhone LiDAR outdoor scans.
Key principle: geometry OVERRIDES ML when shape is unambiguous:
  - vertical + planar + wide  → building (never ground)
  - horizontal + flat         → ground/road/sidewalk
  - non-planar + spread       → vegetation
  - compact + low + non-planar → vehicle
All rules use coordinate-system-independent features (no absolute Z).
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

from ..segments.segment import Segment, SemanticLabel

# index → label  (must match trained model's 8-class taxonomy)
CLASSES: list[SemanticLabel] = [
    "unlabeled",   # 0
    "ground",      # 1
    "road",        # 2
    "sidewalk",    # 3
    "building",    # 4
    "fence",       # 5
    "vegetation",  # 6
    "vehicle",     # 7
]
IDX = {c: i for i, c in enumerate(CLASSES)}


def _segment_votes(probs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Mean softmax over a segment's points → (C,)."""
    return probs[indices].mean(axis=0)


def _color_prior(rgb: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    """Color-based likelihood multipliers."""
    if rgb is None or len(indices) == 0:
        return {}

    seg_rgb = rgb[indices]
    mean_r, mean_g, mean_b = seg_rgb.mean(axis=0)
    priors = {}

    # Green-dominant → vegetation
    if mean_g > mean_r and mean_g > mean_b and mean_g > 0.35:
        greenness = (mean_g - max(mean_r, mean_b))
        priors["vegetation"] = 1.0 + greenness * 8.0
        priors["building"] = max(0.2, 1.0 - greenness * 5.0)
        priors["vehicle"] = max(0.3, 1.0 - greenness * 3.0)

    # Gray/beige (concrete, plaster) → building material
    spread = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_r - mean_b))
    brightness = (mean_r + mean_g + mean_b) / 3.0
    if spread < 0.12 and brightness > 0.35:
        priors["building"] = priors.get("building", 1.0) * 1.8
        priors["vegetation"] = priors.get("vegetation", 1.0) * 0.3

    return priors


def _apply_geom_prior(seg: Segment, scores: np.ndarray,
                      rgb: np.ndarray | None = None) -> np.ndarray:
    """Nudge scores using geometry-derived priors and vetoes."""
    s = scores.copy()
    f = seg.features

    horiz_extent = max(f.extent[0], f.extent[1])
    min_horiz    = min(f.extent[0], f.extent[1])
    area_proxy   = float(f.extent[0] * f.extent[1])

    # ── Color priors ────────────────────────────────────────────────
    if rgb is not None:
        cp = _color_prior(rgb, seg.indices)
        for cls_name, mult in cp.items():
            if cls_name in IDX:
                s[IDX[cls_name]] *= mult

    # ══════════════════════════════════════════════════════════════════
    # CORE RULES — physics-based, apply to ALL segment kinds.
    # ══════════════════════════════════════════════════════════════════

    # Rule 1: Vertical things cannot be ground/road/sidewalk.
    if f.verticality > 0.4 and f.height_range > 0.5:
        s[IDX["ground"]] *= 0.05
        s[IDX["road"]] *= 0.05
        s[IDX["sidewalk"]] *= 0.05

    # Rule 1b: Tall things (>1.5m height range) cannot be ground, even at
    # moderate verticality. Ground is flat by definition.
    if f.height_range > 1.5:
        s[IDX["ground"]] *= 0.1
        s[IDX["road"]] *= 0.1

    # Rule 2: Very tall segments (>3m) are not vehicles (cars are ~1.5m tall).
    if f.height_range > 3.0:
        s[IDX["vehicle"]] *= 0.05

    # Rule 3: Very wide planar segments are not vehicles.
    if f.planarity > 0.5 and horiz_extent > 4.0:
        s[IDX["vehicle"]] *= 0.05

    # ── Planes ──────────────────────────────────────────────────────
    if seg.kind == "plane":

        # Planes are flat surfaces — vehicles are NOT planar.
        s[IDX["vehicle"]] *= 0.1

        if f.verticality > 0.6:
            # Large vertical planar → BUILDING
            if f.planarity > 0.3 and horiz_extent > 1.0:
                s[IDX["building"]] *= 4.0
                s[IDX["vegetation"]] *= 0.2
                s[IDX["ground"]] *= 0.0
                s[IDX["road"]] *= 0.0
                s[IDX["sidewalk"]] *= 0.0

            # Very large wall
            if f.planarity > 0.3 and (horiz_extent > 3.0 or area_proxy > 2.0):
                s[IDX["building"]] *= 5.0
                s[IDX["fence"]] *= 0.3

            # Medium vertical, short → fence
            if f.planarity > 0.3 and horiz_extent > 0.5 and f.height_range < 2.0:
                s[IDX["fence"]] *= 1.5

            # Tall vertical (>2.5m) → building, not fence
            if f.height_range > 2.5 and f.planarity > 0.3:
                s[IDX["building"]] *= 2.0
                s[IDX["fence"]] *= 0.3

            # Any vertical plane: suppress horizontal classes
            s[IDX["ground"]] *= 0.0
            s[IDX["road"]] *= 0.0
            s[IDX["sidewalk"]] *= 0.0

        # Horizontal planes
        if f.verticality < 0.3:
            s[IDX["building"]] *= 0.0
            s[IDX["fence"]] *= 0.0
            s[IDX["vehicle"]] *= 0.0
            s[IDX["ground"]] *= 1.5
            s[IDX["road"]] *= 1.3
            s[IDX["sidewalk"]] *= 1.3

        # Sloped planes (roof) — verticality 0.3–0.6
        if 0.3 <= f.verticality <= 0.6:
            if f.planarity > 0.3 and horiz_extent > 1.5:
                s[IDX["building"]] *= 3.0
                s[IDX["fence"]] *= 0.2
                s[IDX["ground"]] *= 0.1

        # Tiny planes
        if max(f.extent) < 0.4:
            s[IDX["building"]] *= 0.1
            s[IDX["fence"]] *= 0.3

    # ── Ground-origin segments ──────────────────────────────────────
    if seg.kind == "ground":
        # Ground extraction can mis-capture vertical surfaces.
        # Only apply strong ground prior if the segment is actually flat.
        if f.verticality < 0.3 and f.height_range < 2.0:
            # Genuinely flat ground
            s[IDX["ground"]] *= 3.0
            s[IDX["road"]] *= 1.5
            s[IDX["sidewalk"]] *= 1.5
            s[IDX["building"]] *= 0.0
            s[IDX["fence"]] *= 0.0
            s[IDX["vegetation"]] *= 0.3
            s[IDX["vehicle"]] *= 0.0
            s[IDX["unlabeled"]] *= 0.2
        else:
            # "Ground" segment that is actually vertical/tall → likely wall
            s[IDX["ground"]] *= 0.5
            if f.verticality > 0.4:
                s[IDX["building"]] *= 2.0
                s[IDX["ground"]] *= 0.1

    # ── Clusters ────────────────────────────────────────────────────
    if seg.kind == "cluster":

        # ---- VERTICAL clusters → building or fence ----
        if f.verticality > 0.4:
            # Ground/road/sidewalk VETOED for any vertical cluster
            s[IDX["ground"]] *= 0.02
            s[IDX["road"]] *= 0.02
            s[IDX["sidewalk"]] *= 0.02

            # Large + planar → building wall
            if f.planarity > 0.3 and horiz_extent > 1.0:
                s[IDX["building"]] *= 4.0
                s[IDX["vehicle"]] *= 0.1

            # Large even with moderate planarity → building
            if f.n_points > 200 and horiz_extent > 2.0:
                s[IDX["building"]] *= 3.0
                s[IDX["vehicle"]] *= 0.2

            # Very large → almost certainly building
            if f.n_points > 500 or horiz_extent > 3.0 or area_proxy > 4.0:
                s[IDX["building"]] *= 3.0
                s[IDX["vehicle"]] *= 0.05

            # Tall (>2m height range) → building
            if f.height_range > 2.0:
                s[IDX["building"]] *= 2.0
                s[IDX["fence"]] *= 0.3
                s[IDX["vehicle"]] *= 0.2

            # Medium height, moderate size → fence
            if f.height_range < 2.0 and 0.5 < horiz_extent < 4.0 and f.planarity > 0.3:
                s[IDX["fence"]] *= 1.8

        # ---- HORIZONTAL clusters ----
        if f.verticality < 0.3:
            # Flat cluster → ground/sidewalk fragment
            if f.height_range < 0.3:
                s[IDX["ground"]] *= 1.5
                s[IDX["sidewalk"]] *= 1.5
                s[IDX["building"]] *= 0.0

        # ---- VEHICLE-like ----
        # Compact, non-planar, moderate size, low height range
        # Cars: ~4.5m long × ~1.8m wide × ~1.5m tall
        if (f.planarity < 0.4
                and 1.0 < horiz_extent < 6.0
                and 0.5 < min_horiz < 3.0
                and 0.8 < f.height_range < 2.5
                and f.n_points > 100):
            s[IDX["vehicle"]] *= 3.0
            s[IDX["building"]] *= 0.3
            s[IDX["vegetation"]] *= 0.5
            s[IDX["fence"]] *= 0.2

        # ---- VEGETATION-like ----
        # Non-planar, spread → vegetation (not building!)
        if f.planarity < 0.3 and f.sphericity > 0.1:
            s[IDX["vegetation"]] *= 2.5
            s[IDX["building"]] *= 0.3
            s[IDX["fence"]] *= 0.3

        # Low verticality cluster → not a wall
        if f.verticality < 0.3:
            s[IDX["building"]] *= 0.3
            s[IDX["vegetation"]] *= 1.5

        # Short, compact blob → bush/vegetation
        if f.height_range < 0.5 and max(f.extent[:2]) < 1.0 and f.planarity < 0.3:
            s[IDX["building"]] *= 0.0
            s[IDX["fence"]] *= 0.0
            s[IDX["vegetation"]] *= 1.5

    # ── Renormalize ─────────────────────────────────────────────────
    s = np.maximum(s, 0)
    if s.sum() > 0:
        s = s / s.sum()
    return s


def fuse(segments: Iterable[Segment],
         probs: np.ndarray | None,
         rgb: np.ndarray | None = None) -> list[Segment]:
    """Assign semantic label + confidence to each segment.

    `probs`: per-point softmax (N, C). If None → uniform prior.
    `rgb`: per-point RGB (N, 3) in [0,1] for color-based priors.
    """
    out = []
    for seg in segments:
        if probs is not None:
            ml = _segment_votes(probs, seg.indices)
        else:
            ml = np.ones(len(CLASSES)) / len(CLASSES)
        fused = _apply_geom_prior(seg, ml, rgb=rgb)
        top = int(np.argmax(fused))
        seg.semantic = CLASSES[top]
        agreement = float(1.0 - 0.5 * np.abs(ml - fused).sum())
        seg.confidence = float(fused[top] * max(0.0, agreement))
        out.append(seg)
    return out
