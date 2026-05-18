"""Test trained model on an iPhone LiDAR scan.

Usage:
    python test_scan.py --ply scan.ply --weights runs/rtx_a4000_full/best.pt
    python test_scan.py --ply scan.ply --weights runs/rtx_a4000_full/best.pt --voxel 0.5
    python test_scan.py --ply scan.ply --weights runs/rtx_a4000_full/best.pt --export results.ply
"""
import argparse
import numpy as np
import open3d as o3d
import torch


NUM_CLASSES = 8

CLASS_NAMES = [
    "unlabeled",   # 0
    "ground",      # 1
    "road",        # 2
    "sidewalk",    # 3
    "building",    # 4
    "fence",       # 5
    "vegetation",  # 6
    "vehicle",     # 7
]

CLASS_COLORS = {
    0: [0.50, 0.50, 0.50],   # unlabeled  — gray
    1: [0.60, 0.40, 0.20],   # ground     — brown
    2: [0.25, 0.25, 0.25],   # road       — dark gray
    3: [0.70, 0.70, 0.70],   # sidewalk   — light gray
    4: [0.90, 0.20, 0.20],   # building   — red
    5: [0.90, 0.60, 0.10],   # fence      — orange
    6: [0.10, 0.65, 0.10],   # vegetation — green
    7: [0.20, 0.40, 0.90],   # vehicle    — blue
}


def main():
    ap = argparse.ArgumentParser(description="Test Okolis AI model on a PLY scan")
    ap.add_argument("--ply", required=True, help="Path to input .ply file")
    ap.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    ap.add_argument("--voxel", type=float, default=1.0,
                    help="Voxel downsample size (smaller = more detail, more RAM)")
    ap.add_argument("--max-points", type=int, default=16384,
                    help="Max points per GPU chunk")
    ap.add_argument("--export", type=str, default=None,
                    help="Export colorized result as PLY file")
    args = ap.parse_args()

    # 1. Load and downsample
    print(f"Loading {args.ply}...")
    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"  Original: {len(pcd.points)} points")

    pcd = pcd.voxel_down_sample(args.voxel)
    print(f"  After voxel {args.voxel}: {len(pcd.points)} points")

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros((len(xyz), 3), dtype=np.float32)

    # Centre the cloud
    xyz_centered = xyz - xyz.mean(axis=0)

    # 2. Compute height above ground (lowest 10% = pseudo-ground)
    z_thresh = np.percentile(xyz_centered[:, 2], 10)
    hag = np.clip(xyz_centered[:, 2] - z_thresh, 0, None).astype(np.float32)

    # 3. Pack features: [R, G, B, intensity(=0), height_above_ground]
    feats = np.zeros((len(xyz_centered), 5), dtype=np.float32)
    feats[:, 0:3] = rgb
    feats[:, 3] = 0.0   # no intensity from iPhone
    feats[:, 4] = hag

    # 4. Load model
    print(f"Loading model {args.weights}...")
    from okolis_ai.ml.randlanet.model import RandLANet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)

    model_cfg = checkpoint.get("cfg", {})
    num_classes = model_cfg.get("num_classes", NUM_CLASSES)
    # Extract PTv3 architecture config from checkpoint
    ptv3_cfg = model_cfg.get("ptv3", {})
    model_kwargs = dict(
        in_feat_dim=model_cfg.get("in_feat_dim", 5),
        num_classes=num_classes,
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

    model = RandLANet(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 5. Chunked inference
    N = len(xyz_centered)
    all_probs = np.zeros((N, num_classes), dtype=np.float32)
    chunk = args.max_points

    print(f"Running inference on {N} points (chunks of {chunk}, device={device})...")

    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            n = end - start
            if n < 64:
                continue

            sub_xyz = xyz_centered[start:end].copy()
            sub_feat = feats[start:end].copy()

            # Pad if chunk is too small
            if n < chunk:
                pad = chunk - n
                sub_xyz = np.vstack([sub_xyz, np.tile(sub_xyz[-1:], (pad, 1))])
                sub_feat = np.vstack([sub_feat, np.tile(sub_feat[-1:], (pad, 1))])

            xyz_t = torch.from_numpy(sub_xyz).float().unsqueeze(0).to(device)
            feat_t = torch.from_numpy(sub_feat).float().unsqueeze(0).to(device)
            feat_t = torch.nan_to_num(feat_t, nan=0.0)

            logits = model(xyz_t, feat_t)  # (1, chunk, C)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs[start:end] = probs[:n]

            # Progress
            done = min(end, N)
            pct = 100 * done / N
            print(f"  {done}/{N} ({pct:.0f}%)", end="\r")

    print()

    # 6. Results
    preds = all_probs.argmax(axis=1)
    confidence = all_probs.max(axis=1)

    print("\n=== Segmentation Results ===")
    print(f"{'Class':<15} {'Count':>7} {'%':>6}  {'Avg Conf':>8}")
    print("-" * 42)
    for c in range(num_classes):
        mask = preds == c
        count = mask.sum()
        pct = 100 * count / N
        avg_conf = float(confidence[mask].mean()) * 100 if count > 0 else 0.0
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {name:<13} {count:>7}  {pct:>5.1f}%  {avg_conf:>7.1f}%")

    # 7. Colorize
    colors = np.array([CLASS_COLORS.get(p, [1, 1, 1]) for p in preds], dtype=np.float64)

    # 8. Export if requested
    if args.export:
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(xyz)
        pcd_out.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(args.export, pcd_out)
        print(f"\nExported: {args.export}")

    # 9. Show viewer
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(xyz)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)

    print("\nOpening viewer (close window to exit)...")
    print("Colors: brown=ground, dark-gray=road, light-gray=sidewalk,")
    print("        red=building, orange=fence, green=vegetation, blue=vehicle")
    o3d.visualization.draw_geometries([pcd_vis], window_name="Okolis AI - Segmentation")


if __name__ == "__main__":
    main()