"""End-to-end test: kod_Tina.ply → potpuna segmentacija kroz sve OkolisAI module.

Koristi PRAVI pipeline — sve ide kroz produkcijske module, ništa inline:
  1. ply_loader.py       → učitavanje PLY
  2. preprocess.py       → outlier removal + gravity align + downsample + normali
  3. ground.py           → ground extraction (grid)
  4. planes.py           → RANSAC plane detection
  5. clusters.py         → DBSCAN clustering
  6. features.py         → geometrijske značajke segmenata
  7. segmenter.py        → PTv3 ML model (wrappan u RandLANetSegmenter)
  8. inference.py         → tiled inference s overlapom
  9. hybrid.py           → fusion geometrija + ML → finalne labele

Pokreni:
    cd okolisAI-project
    python test_pipeline.py

Potrebno:
    pip install torch open3d numpy scipy
"""
from __future__ import annotations
import sys
import os
import time
import numpy as np

# Add project root to path so OkolisAI modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ──────────────────────────────────────────────────────────────
PLY_PATH   = "kod_Tina.ply"
WEIGHTS    = "PTv3(1).pt"
VOXEL      = 0.05          # downsample voxel size (metres) — 5cm good for iPhone
EXPORT_PLY = "kod_Tina_segmented.ply"   # coloured output

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
    import open3d as o3d
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    total_t0 = time.time()

    # ── 1. UČITAVANJE (ply_loader.py) ───────────────────────────────────
    print(f"\n[1/7] Učitavam {PLY_PATH} (ply_loader.py)...")
    t0 = time.time()
    from OkolisAI.io.ply_loader import load_ply
    pcd = load_ply(PLY_PATH)
    n_orig = len(pcd.points)
    print(f"  Original: {n_orig:,} točaka")
    print(f"  Vrijeme: {time.time()-t0:.1f}s")

    # ── 2. PREDOBRADA (preprocess.py) ───────────────────────────────────
    #   - outlier removal (statistical + radius)
    #   - gravity alignment (RANSAC na donjem dijelu → Z=gore)
    #   - voxel downsample
    #   - normal estimation
    print(f"\n[2/7] Predobrada (preprocess.py)...")
    t0 = time.time()
    from OkolisAI.geometry.preprocess import (
        remove_outliers, gravity_align, voxel_downsample, estimate_normals
    )

    print(f"  Outlier removal...")
    pcd = remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0, radius=0.10, min_nb=5)
    print(f"    Nakon outlier removal: {len(pcd.points):,} točaka")

    print(f"  Gravity alignment...")
    pcd, R_gravity = gravity_align(pcd)
    is_aligned = not np.allclose(R_gravity, np.eye(3))
    print(f"    {'Poravnano' if is_aligned else 'Već poravnano (identity R)'}")

    print(f"  Voxel downsample ({VOXEL}m)...")
    pcd = voxel_downsample(pcd, voxel=VOXEL)
    print(f"    Nakon downsample: {len(pcd.points):,} točaka")

    print(f"  Normal estimation...")
    pcd = estimate_normals(pcd, radius=max(0.15, VOXEL * 5), k=30,
                           viewpoint=np.array([0, 0, 100.0]))
    print(f"  Vrijeme: {time.time()-t0:.1f}s")

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = (np.asarray(pcd.colors, dtype=np.float32)
           if pcd.has_colors()
           else np.zeros((len(xyz), 3), dtype=np.float32))
    N = len(xyz)

    # ── 3. GEOMETRIJA ──────────────────────────────────────────────────

    # 3a. Ground extraction (ground.py)
    print(f"\n[3/7] Geometrija...")
    t0 = time.time()
    from OkolisAI.geometry.ground import extract_ground, height_above_ground
    print(f"  Ground extraction (grid metoda)...")
    ground_mask = extract_ground(pcd, method="grid", cell=0.3, z_tol=0.15)
    h_above = height_above_ground(pcd, ground_mask, cell=0.3)
    ground_idx = np.where(ground_mask)[0]
    print(f"    Ground točaka: {len(ground_idx):,} ({100*len(ground_idx)/N:.1f}%)")

    # 3b. Plane detection (planes.py)
    from OkolisAI.geometry.planes import extract_planes
    print(f"  Plane detection (RANSAC)...")
    planes = extract_planes(pcd, exclude=ground_idx,
                            distance=VOXEL * 1.5,
                            min_inliers=max(50, N // 100),
                            max_planes=30)
    plane_pts = sum(len(p.indices) for p in planes)
    print(f"    Ravnina: {len(planes)}, točaka: {plane_pts:,} ({100*plane_pts/N:.1f}%)")

    # 3c. Clustering (clusters.py)
    from OkolisAI.geometry.clusters import cluster_residual
    print(f"  Clustering (DBSCAN)...")
    used = np.zeros(N, dtype=bool)
    used[ground_idx] = True
    for p in planes:
        used[p.indices] = True
    remaining = np.where(~used)[0]
    clusters = cluster_residual(pcd, remaining, eps=VOXEL * 5, min_points=20)
    cluster_pts = sum(len(c.indices) for c in clusters)
    print(f"    Preostalo: {len(remaining):,}, klastera: {len(clusters)}, "
          f"u klasterima: {cluster_pts:,}")

    # 3d. Segment features (features.py)
    from OkolisAI.geometry import features as feat_mod
    from OkolisAI.segments.segment import Segment
    print(f"  Kreiranje segmenata + features...")
    segments = []
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
    print(f"    Segmenata: {len(segments)} "
          f"(1 ground + {len(planes)} planes + {len(clusters)} clusters)")
    print(f"  Vrijeme: {time.time()-t0:.1f}s")

    # ── 4. ML SEGMENTACIJA (segmenter.py + inference.py) ────────────────
    #   Koristi RandLANetSegmenter (wrapa PTv3 model) +
    #   inference.py tiled s overlapom za stabilne granice
    print(f"\n[4/7] ML segmentacija (segmenter.py + inference.py, tiled s overlapom)...")
    t0 = time.time()
    from OkolisAI.okolis_ai.datasets.common import pack_features
    from OkolisAI.ml.randlanet.segmenter import RandLANetSegmenter
    from OkolisAI.ml.inference import segment_cloud

    feats = pack_features(rgb=rgb, intensity=None, height_above_ground=h_above)

    # Centre the cloud (model trained on centred data)
    xyz_c = (xyz - xyz.mean(axis=0)).astype(np.float32)

    print(f"  Učitavam model: {WEIGHTS}")
    segmenter = RandLANetSegmenter(weights=WEIGHTS, device=device)
    print(f"    Klasa: {segmenter.num_classes}, device: {segmenter.device}")

    print(f"  Tiled inference ({N:,} točaka, tile=15m, overlap=10%)...")
    all_probs = segment_cloud(
        segmenter, xyz_c, features=feats,
        tile_xy=15.0, overlap=0.1,
    )
    print(f"  Vrijeme: {time.time()-t0:.1f}s")

    # ── 4b. ML-only rezultati (prije fusion-a) ─────────────────────────
    ml_preds = all_probs.argmax(axis=1)
    ml_conf = all_probs.max(axis=1)

    print(f"\n  === ML-Only Rezultati (bez fusion-a) ===")
    print(f"  {'Klasa':<15} {'Broj':>8} {'%':>6}  {'Avg Conf':>8}")
    print(f"  {'-'*42}")
    for c in range(segmenter.num_classes):
        mask = ml_preds == c
        count = mask.sum()
        pct = 100 * count / N
        avg_conf = float(ml_conf[mask].mean()) * 100 if count > 0 else 0.0
        print(f"  {CLASS_NAMES[c]:<13} {count:>8,}  {pct:>5.1f}%  {avg_conf:>7.1f}%")

    # ── 5. FUZIJA (hybrid.py) ──────────────────────────────────────────
    print(f"\n[5/7] Fusion (hybrid.py — geometrija + ML)...")
    t0 = time.time()
    from OkolisAI.fusion.hybrid import fuse, CLASSES
    segments = fuse(segments, all_probs, rgb=rgb)

    print(f"\n  === Fusion Rezultati (geometrija + ML) ===")
    print(f"  {'Segment':<20} {'Tip':<10} {'Label':<12} {'Conf':>6} {'Pts':>8}")
    print(f"  {'-'*60}")

    label_counts = {}
    for s in segments:
        label_counts[s.semantic] = label_counts.get(s.semantic, 0) + len(s.indices)
        if len(s.indices) >= 100:  # show only meaningful segments
            print(f"  {s.id:<20} {s.kind:<10} {s.semantic:<12} "
                  f"{s.confidence:>5.1%} {len(s.indices):>8,}")

    print(f"\n  === Sažetak po klasama (fusion) ===")
    print(f"  {'Klasa':<15} {'Točaka':>10} {'%':>6}")
    print(f"  {'-'*35}")
    total_segmented = sum(label_counts.values())
    for label in CLASSES:
        count = label_counts.get(label, 0)
        pct = 100 * count / N if N > 0 else 0
        print(f"  {label:<13} {count:>10,}  {pct:>5.1f}%")
    unseg = N - total_segmented
    if unseg > 0:
        print(f"  {'(neseg.)':<13} {unseg:>10,}  {100*unseg/N:>5.1f}%")

    print(f"  Vrijeme: {time.time()-t0:.1f}s")

    # ── 6. EXPORT ──────────────────────────────────────────────────────
    print(f"\n[6/7] Export segmentiranih PLY datoteka...")
    from OkolisAI.io.ply_loader import save_ply, from_numpy

    # Fusion result (geometry + ML)
    colors_fusion = np.tile(np.array(CLASS_COLORS[0], dtype=np.float64), (N, 1))
    for s in segments:
        label_idx = CLASSES.index(s.semantic) if s.semantic in CLASSES else 0
        colors_fusion[s.indices] = CLASS_COLORS[label_idx]

    pcd_fusion = from_numpy(xyz, rgb=colors_fusion)
    save_ply(pcd_fusion, EXPORT_PLY)
    size_mb = os.path.getsize(EXPORT_PLY) / 1e6
    print(f"  Fusion: {EXPORT_PLY} ({size_mb:.1f} MB)")

    # ML-only result for comparison
    ml_export = "kod_Tina_ml_only.ply"
    ml_colors = np.array([CLASS_COLORS.get(p, [1, 1, 1]) for p in ml_preds],
                         dtype=np.float64)
    pcd_ml = from_numpy(xyz, rgb=ml_colors)
    save_ply(pcd_ml, ml_export)
    print(f"  ML-only: {ml_export} ({os.path.getsize(ml_export)/1e6:.1f} MB)")

    # ── 7. USPOREDBA ──────────────────────────────────────────────────
    total_time = time.time() - total_t0
    print(f"\n[7/7] Usporedba ML vs Fusion:")
    print(f"  {'Klasa':<15} {'ML':>8} {'Fusion':>8} {'Razlika':>8}")
    print(f"  {'-'*42}")
    for c in range(len(CLASS_NAMES)):
        ml_count = int((ml_preds == c).sum())
        fusion_count = label_counts.get(CLASS_NAMES[c], 0)
        diff = fusion_count - ml_count
        sign = "+" if diff > 0 else ""
        print(f"  {CLASS_NAMES[c]:<13} {ml_count:>8,} {fusion_count:>8,} {sign}{diff:>7,}")

    print(f"\n{'='*55}")
    print(f"GOTOVO! Ukupno vrijeme: {total_time:.1f}s")
    print(f"\nPipeline korišten (svi pravi moduli):")
    print(f"  ply_loader → preprocess (outlier+gravity+downsample+normals)")
    print(f"  → ground → planes → clusters → features")
    print(f"  → segmenter+inference (PTv3, tiled 15m, 10% overlap)")
    print(f"  → fusion (geometry priors + ML + color priors)")
    print(f"\nOtvori PLY datoteke u MeshLab ili Open3D:")
    print(f"  {EXPORT_PLY}      — fusion rezultat")
    print(f"  {ml_export}    — samo ML predikcija")
    print(f"\nBoje: smeđa=ground, tamno-siva=road, svijetlo-siva=sidewalk,")
    print(f"      crvena=building, narančasta=fence, zelena=vegetation, plava=vehicle")

    # Vizualizacija (neće raditi ako nema display-a)
    try:
        import open3d as o3d
        pcd_ml_vis = o3d.io.read_point_cloud(ml_export)
        o3d.visualization.draw_geometries([pcd_ml_vis], window_name="ML Only")
        pcd_fused_vis = o3d.io.read_point_cloud(EXPORT_PLY)
        o3d.visualization.draw_geometries([pcd_fused_vis], window_name="Fusion (geo + ML)")
    except Exception:
        print("\n(Vizualizacija preskočena — nema GUI display-a)")


if __name__ == "__main__":
    main()
