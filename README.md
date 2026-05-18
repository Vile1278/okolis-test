# Okoliš AI — Test Pipeline

End-to-end test za Okoliš AI segmentaciju point cloudova.  
Koristi Point Transformer V3 model treniran na Toronto-3D + SemanticKITTI + Pandaset.

## Pipeline

```
PLY scan → preprocess (outlier removal + gravity align + downsample + normals)
         → ground extraction (grid)
         → plane detection (RANSAC)
         → clustering (DBSCAN)
         → segment features (PCA)
         → ML segmentation (PTv3, tiled 15m, 10% overlap)
         → fusion (geometry priors + ML + color priors)
         → export segmented PLY
```

## Klase (8)

| ID | Klasa     | Boja         |
|----|-----------|--------------|
| 0  | unlabeled | siva         |
| 1  | ground    | smeđa        |
| 2  | road      | tamno-siva   |
| 3  | sidewalk  | svijetlo-siva|
| 4  | building  | crvena       |
| 5  | fence     | narančasta   |
| 6  | vegetation| zelena       |
| 7  | vehicle   | plava        |

## Setup

```bash
pip install -r requirements.txt
```

## Potrebne datoteke (nisu u repo-u — prevelike za GitHub)

- `PTv3(1).pt` — trenirani PTv3 weightsovi (~88 MB)
- `kod_Tina.ply` — iPhone LiDAR sken za testiranje (~312 MB)

## Pokretanje

```bash
python test_pipeline.py
```

Izlaz: `kod_Tina_segmented.ply` (fusion) i `kod_Tina_ml_only.ply` (samo ML).  
Otvori u MeshLab ili Open3D za vizualizaciju.
