"""Hybrid geometry + ML label assignment at the segment level.

Geometry priors tuned for iPhone LiDAR outdoor scans.
Key principle: geometry NUDGES ML predictions, doesn't override them.
Hard vetoes (×0.0) only when physics makes a class impossible.
Multipliers are moderate (max ~3×) and don't compound excessively.

Rules:
  - vertical + planar + wide  → building boost
  - horizontal + flat         → ground/road/sidewalk boost
  - non-planar + spread       → vegetation boost
  - compact + low + non-planar → vehicle boost
  - ground-extracted segments  → trust the extractor
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
        priors["vegetation"] = 1.0 + greenness * 5.0
        priors["building"] = max(0.4, 1.0 - greenness * 3.0)
        priors["vehicle"] = max(0.5, 1.0 - greenness * 2.0)

    # Gray/beige (concrete, plaster) → mild building boost
    spread = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_r - mean_b))
    brightness = (mean_r + mean_g + mean_b) / 3.0
    if spread < 0.12 and brightness > 0.35:
        priors["building"] = priors.get("building", 1.0) * 1.3
        priors["vegetation"] = priors.get("vegetation", 1.0) * 0.5

    # Brown/dark → ground hint
    if mean_r > mean_g and mean_r > mean_b and brightness < 0.4:
        priors["ground"] = priors.get("ground", 1.0) * 1.3

    return priors


def _apply_geom_prior(seg: Segment, scores: np.ndarray,
                      rgb: np.ndarray | None = None) -> np.ndarray:
    """Nudge scores using geometry-derived priors and vetoes."""
    s = scores.copy()
    f = seg.features

    horiz_extent = max(f.extent[0], f.extent[1])
    min_horiz    = min(f.extent[0], f.extent[1])

    # ── Color priors ────────────────────────────────────────────────
    if rgb is not None:
        cp = _color_prior(rgb, seg.indices)
        for cls_name, mult in cp.items():
            if cls_name in IDX:
                s[IDX[cls_name]] *= mult

    # ══════════════════════════════════════════════════════════════════
    # CORE RULES — physics-based, apply to non-ground segment kinds.
    # Ground segments are handled separately (trust the extractor).
    # ══════════════════════════════════════════════════════════════════

    if seg.kind != "ground":
        # Rule 1: Vertical things cannot be ground/road/sidewalk.
        if f.verticality > 0.5 and f.height_range > 0.8:
            s[IDX["ground"]] *= 0.1
            s[IDX["road"]] *= 0.1
            s[IDX["sidewalk"]] *= 0.1

        # Rule 2: Very tall segments (>3m) are not vehicles.
        if f.height_range > 3.0:
            s[IDX["vehicle"]] *= 0.1

        # Rule 3: Very wide planar segments are not vehicles.
        if f.planarity > 0.5 and horiz_extent > 4.0:
            s[IDX["vehicle"]] *= 0.1

    # ── Ground-origin segments ──────────────────────────────────────
    # Trust the ground extractor. It already handles slopes, so
    # height_range can be large on hilly terrain — that's OK.
    if seg.kind == "ground":
        if f.verticality < 0.5:
            # Ground extractor says ground + not vertical → believe it.
            # Let ML decide between ground/road/sidewalk.
            s[IDX["ground"]] *= 2.0
            s[IDX["road"]] *= 1.5
            s[IDX["sidewalk"]] *= 1.5
            s[IDX["building"]] *= 0.05
            s[IDX["fence"]] *= 0.05
            s[IDX["vehicle"]] *= 0.05
            s[IDX["unlabeled"]] *= 0.2
            # Vegetation can grow on ground — mild suppress only
            s[IDX["vegetation"]] *= 0.5
        else:
            # "Ground" segment but actually vertical — extractor error.
            # Don't force ground; let ML decide, mild building boost.
            s[IDX["building"]] *= 1.5
            s[IDX["ground"]] *= 0.5

    # ── Planes ──────────────────────────────────────────────────────
    if seg.kind == "plane":

        # Planes are flat surfaces — vehicles are NOT planar.
        s[IDX["vehicle"]] *= 0.2

        if f.verticality > 0.6:
            # Vertical plane → building or fence, not ground
            s[IDX["ground"]] *= 0.05
            s[IDX["road"]] *= 0.05
            s[IDX["sidewalk"]] *= 0.05

            # Large vertical planar → building
            if f.planarity > 0.3 and horiz_extent > 1.0:
                s[IDX["building"]] *= 3.0
                s[IDX["vegetation"]] *= 0.3

            # Tall vertical (>2.5m) → building, not fence
            if f.height_range > 2.5 and f.planarity > 0.3:
                s[IDX["building"]] *= 1.5
                s[IDX["fence"]] *= 0.4

            # Short vertical (<2m) → could be fence
            if f.height_range < 2.0 and f.planarity > 0.3 and horiz_extent > 0.5:
                s[IDX["fence"]] *= 1.5

        # Horizontal planes → ground/road/sidewalk
        if f.verticality < 0.3:
            s[IDX["building"]] *= 0.1
            s[IDX["fence"]] *= 0.1
            s[IDX["vehicle"]] *= 0.1
            s[IDX["ground"]] *= 1.5
            s[IDX["road"]] *= 1.3
            s[IDX["sidewalk"]] *= 1.3

        # Sloped planes (roof) — verticality 0.3–0.6
        if 0.3 <= f.verticality <= 0.6:
            if f.planarity > 0.3 and horiz_extent > 1.5:
                s[IDX["building"]] *= 2.0
                s[IDX["fence"]] *= 0.3
                s[IDX["ground"]] *= 0.2

        # Tiny planes — don't trust shape
        if max(f.extent) < 0.4:
            s[IDX["building"]] *= 0.3
            s[IDX["fence"]] *= 0.5

    # ── Clusters ────────────────────────────────────────────────────
    if seg.kind == "cluster":

        # ---- VERTICAL clusters ----
        if f.verticality > 0.4:
            # Suppress horizontal classes
            s[IDX["ground"]] *= 0.1
            s[IDX["road"]] *= 0.1
            s[IDX["sidewalk"]] *= 0.1

            # Large + planar → building wall (single boost, no stacking)
            if f.planarity > 0.3 and (horiz_extent > 2.0 or f.n_points > 500):
                s[IDX["building"]] *= 3.0
                s[IDX["vehicle"]] *= 0.2
            elif f.planarity > 0.3 and horiz_extent > 1.0:
                s[IDX["building"]] *= 2.0
                s[IDX["vehicle"]] *= 0.3

            # Tall (>2.5m height range) → building rather than fence
            if f.height_range > 2.5:
                s[IDX["building"]] *= 1.5
                s[IDX["fence"]] *= 0.4

            # Medium height, moderate size → fence
            if f.height_range < 2.0 and 0.5 < horiz_extent < 4.0 and f.planarity > 0.3:
                s[IDX["fence"]] *= 1.8

        # ---- HORIZONTAL clusters ----
        if f.verticality < 0.3:
            # Flat cluster → ground/sidewalk fragment, not building
            s[IDX["building"]] *= 0.3
            s[IDX["vegetation"]] *= 1.3
            if f.height_range < 0.3:
                s[IDX["ground"]] *= 1.5
                s[IDX["sidewalk"]] *= 1.5

        # ---- VEHICLE-like ----
        # Compact, non-planar, moderate size, low height range
        # Cars: ~4.5m long × ~1.8m wide × ~1.5m tall
        if (f.planarity < 0.4
                and 1.0 < horiz_extent < 6.0
                and 0.5 < min_horiz < 3.0
                and 0.8 < f.height_range < 2.5
                and f.n_points > 100):
            s[IDX["vehicle"]] *= 2.5
            s[IDX["building"]] *= 0.4
            s[IDX["vegetation"]] *= 0.6
            s[IDX["fence"]] *= 0.3

        # ---- VEGETATION-like ----
        # Non-planar, spread → vegetation (not building!)
        if f.planarity < 0.3 and f.sphericity > 0.1:
            s[IDX["vegetation"]] *= 2.0
            s[IDX["building"]] *= 0.5
            s[IDX["fence"]] *= 0.5

        # Short, compact blob → bush/vegetation
        if f.height_range < 0.5 and max(f.extent[:2]) < 1.0 and f.planarity < 0.3:
            s[IDX["building"]] *= 0.1
            s[IDX["fence"]] *= 0.1
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
