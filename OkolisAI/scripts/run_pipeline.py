"""End-to-end CLI: .ply → segmented, wall-extracted Scene.

Fixed version:
  - Uses Open3D for PLY loading (handles any format)
  - Chunked inference for 6GB GPU
  - pack_features handles None intensity

Examples:
    python -m okolis_ai.scripts.run_pipeline --input scans/yard.ply \
        --output outputs/yard_scene --view
    python -m okolis_ai.scripts.run_pipeline --input scans/yard.ply \
        --output outputs/yard_scene --model weights/randlanet_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from ..geometry.ground import extract_ground, height_above_ground
from ..geometry.planes import extract_planes
from ..geometry.clusters import cluster_residual
from ..geometry import features as feat_mod
from ..segments.segment import Segment
from ..walls.wall_extractor import extract_wall
from ..fusion.hybrid import fuse
from ..ml.base import UniformSegmenter
from ..scene.scene import Scene


def _load_and_preprocess(ply_path, voxel=0.03):
    """Load PLY with Open3D (handles any format) and preprocess."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_orig = len(pcd.points)
    print(f"  Loaded: {n_orig} points")

    if n_orig == 0:
        raise RuntimeError(f"PLY file has 0 points: {ply_path}")

    pcd = pcd.voxel_down_sample(voxel)
    print(f"  After voxel {voxel}: {len(pcd.points)} points")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal: {len(pcd.points)} points")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 100.0]))

    return pcd


def _chunked_segment(segmenter, xyz, feats, max_points=16384):
    """Run ML segmentation in chunks to fit in GPU memory."""
    import torch

    N = len(xyz)
    num_classes = segmenter.num_classes if hasattr(segmenter, 'num_classes') else 8
    all_probs = np.zeros((N, num_classes), dtype=np.float32)

    if hasattr(segmenter, 'model'):
        device = next(segmenter.model.parameters()).device
        segmenter.model.eval()

        with torch.no_grad():
            for start in range(0, N, max_points):
                end = min(start + max_points, N)
                n = end - start
                if n < 64:
                    continue

                sub_xyz = xyz[start:end].copy()
                sub_feat = feats[start:end].copy()

                if n < max_points:
                    pad = max_points - n
                    sub_xyz = np.vstack([sub_xyz, np.tile(sub_xyz[-1:], (pad, 1))])
                    sub_feat = np.vstack([sub_feat, np.tile(sub_feat[-1:], (pad, 1))])

                xyz_t = torch.from_numpy(sub_xyz).float().unsqueeze(0).to(device)
                feat_t = torch.from_numpy(sub_feat).float().unsqueeze(0).to(device)
                feat_t = torch.nan_to_num(feat_t, nan=0.0)

                logits = segmenter.model(xyz_t, feat_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                all_probs[start:end] = probs[:n]
    else:
        all_probs = segmenter.predict(xyz, feats)

    return all_probs


def build_scene(ply_path: Path, model_weights: Path | None = None,
                voxel: float = 0.03, max_points: int = 16384) -> Scene:

    print(f"[pipeline] Loading {ply_path}...")
    pcd = _load_and_preprocess(ply_path, voxel=voxel)

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros((len(xyz), 3), dtype=np.float32)

    ground_mask = extract_ground(pcd, method="grid", cell=0.3, z_tol=0.15)
    h_above = height_above_ground(pcd, ground_mask, cell=0.3)
    ground_idx = np.where(ground_mask)[0]

    planes = extract_planes(pcd, exclude=ground_idx,
                            distance=voxel * 1.5, min_inliers=max(50, len(xyz) // 100),
                            max_planes=30)
    used = np.zeros(len(xyz), dtype=bool); used[ground_idx] = True
    for p in planes: used[p.indices] = True
    remaining = np.where(~used)[0]
    clusters = cluster_residual(pcd, remaining, eps=voxel * 5, min_points=20)

    segments: list[Segment] = []
    if len(ground_idx):
        segments.append(Segment.new(
            kind="ground", indices=ground_idx,
            features=feat_mod.compute(xyz[ground_idx])))
    for p in planes:
        segments.append(Segment.new(
            kind="plane", indices=p.indices,
            features=feat_mod.compute(xyz[p.indices]),
            normal=p.normal, plane=p.plane))
    for c in clusters:
        segments.append(Segment.new(
            kind="cluster", indices=c.indices,
            features=feat_mod.compute(xyz[c.indices])))

    if model_weights is not None:
        from ..ml.randlanet.segmenter import RandLANetSegmenter
        segmenter = RandLANetSegmenter(weights=model_weights)
    else:
        segmenter = UniformSegmenter()

    from ..okolis_ai.datasets.common import pack_features
    feats = pack_features(rgb=rgb, intensity=None, height_above_ground=h_above)

    print(f"[pipeline] Running ML segmentation ({len(xyz)} points, chunks of {max_points})...")
    probs = _chunked_segment(segmenter, xyz, feats, max_points=max_points)

    segments = fuse(segments, probs, rgb=rgb)

    walls = []
    plane_refs = [(p.normal, p.centroid) for p in planes]
    for s in segments:
        if s.semantic == "building" and s.kind == "plane" and s.normal is not None:
            pts = xyz[s.indices]
            try:
                w = extract_wall(pts, s.normal, s.id,
                                 confidence=s.confidence,
                                 all_planes=plane_refs)
                walls.append(w)
            except Exception as e:
                print(f"[warn] wall extraction failed for {s.id}: {e}")

    return Scene(points=xyz, colors=rgb, segments=segments, walls=walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--max-points", type=int, default=16384,
                    help="Max points per GPU chunk")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    scene = build_scene(args.input, args.model, voxel=args.voxel,
                        max_points=args.max_points)
    scene.save(args.output)
    print(f"Saved scene → {args.output}  "
          f"({len(scene.segments)} segments, {len(scene.walls)} walls)")

    if args.view:
        from ..interaction.viewer import show
        show(scene)


if __name__ == "__main__":
    main()