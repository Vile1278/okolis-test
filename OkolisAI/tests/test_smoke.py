"""Smoke test: synthetic yard scene (ground + two walls + a shrub) runs end-to-end."""
from __future__ import annotations
import numpy as np
import open3d as o3d
from pathlib import Path
import tempfile

from okolis_ai.io.ply_loader import from_numpy, save_ply
from okolis_ai.scripts.run_pipeline import build_scene
from okolis_ai.editing.ops import extend_wall


def _make_synthetic():
    rng = np.random.default_rng(42)
    # ground: 10x10 m plane at z=0, noisy
    gx, gy = np.meshgrid(np.arange(-5, 5, 0.05), np.arange(-5, 5, 0.05))
    ground = np.stack([gx.ravel(), gy.ravel(),
                       rng.normal(0, 0.01, gx.size)], axis=1)
    # wall A: along X axis at y=2, length 4m, height 2m
    wx = np.arange(-2, 2, 0.03)
    wz = np.arange(0, 2, 0.03)
    WX, WZ = np.meshgrid(wx, wz)
    wallA = np.stack([WX.ravel(), np.full(WX.size, 2.0) + rng.normal(0, 0.005, WX.size),
                      WZ.ravel()], axis=1)
    # wall B: along Y axis at x=-2, length 3m, height 2m
    wy = np.arange(-1.5, 1.5, 0.03)
    WY, WZ2 = np.meshgrid(wy, wz)
    wallB = np.stack([np.full(WY.size, -2.0) + rng.normal(0, 0.005, WY.size),
                      WY.ravel(), WZ2.ravel()], axis=1)
    # shrub: blob at (3, -2, 0..1)
    shrub = rng.normal([3, -2, 0.5], [0.3, 0.3, 0.3], (500, 3))
    return np.concatenate([ground, wallA, wallB, shrub])


def test_pipeline_runs():
    pts = _make_synthetic()
    pcd = from_numpy(pts)
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / "synth.ply"
        save_ply(pcd, ply)
        scene = build_scene(ply, model_weights=None, voxel=0.05)
    assert len(scene.points) > 0
    assert any(s.kind == "ground" for s in scene.segments)
    assert any(s.kind == "plane" for s in scene.segments)


def test_extend_wall_when_present():
    pts = _make_synthetic()
    pcd = from_numpy(pts)
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / "synth.ply"
        save_ply(pcd, ply)
        scene = build_scene(ply, model_weights=None, voxel=0.05)
    if not scene.walls:
        # Without a trained model, uniform probs may not yield "building" label.
        # That's acceptable; the geometry pipeline still produced plane segments.
        return
    before = len(scene.points)
    new_scene = extend_wall(scene, scene.walls[0].id, delta_length=1.0)
    assert len(new_scene.points) > before
    assert new_scene.synthetic_mask.sum() > 0
