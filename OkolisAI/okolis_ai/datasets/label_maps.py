"""Unified label taxonomy and per-dataset remaps.

Target classes (must match trained model — 8 classes):
    0 unlabeled | 1 ground | 2 road | 3 sidewalk | 4 building | 5 fence | 6 vegetation | 7 vehicle
"""
NUM_CLASSES = 8

UNIFIED = {
    "unlabeled": 0,
    "ground": 1,
    "road": 2,
    "sidewalk": 3,
    "building": 4,
    "fence": 5,
    "vegetation": 6,
    "vehicle": 7,
}

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

# ---- Toronto-3D (8 classes in original) ----
# 0 unclassified, 1 road, 2 road marking, 3 natural, 4 building,
# 5 utility line, 6 pole, 7 car, 8 fence
TORONTO3D_MAP = {
    0: UNIFIED["unlabeled"],
    1: UNIFIED["road"],
    2: UNIFIED["road"],        # road marking → road
    3: UNIFIED["vegetation"],  # natural → vegetation
    4: UNIFIED["building"],    # building
    5: UNIFIED["unlabeled"],   # utility line → unlabeled (no pole class)
    6: UNIFIED["unlabeled"],   # pole → unlabeled
    7: UNIFIED["vehicle"],     # car → vehicle
    8: UNIFIED["fence"],       # fence
}

# ---- SemanticKITTI (20 classes learning set) ----
SEMKITTI_MAP = {
    0:  UNIFIED["unlabeled"],
    9:  UNIFIED["road"],       # road
    10: UNIFIED["sidewalk"],   # sidewalk
    11: UNIFIED["ground"],     # other-ground
    13: UNIFIED["building"],   # building
    14: UNIFIED["fence"],      # fence
    15: UNIFIED["vegetation"], # vegetation
    16: UNIFIED["vegetation"], # trunk → vegetation
    17: UNIFIED["ground"],     # terrain → ground
    18: UNIFIED["unlabeled"],  # pole → unlabeled
    19: UNIFIED["unlabeled"],  # traffic sign → unlabeled
    1:  UNIFIED["vehicle"],    # car → vehicle
    2:  UNIFIED["vehicle"],    # bicycle → vehicle
    3:  UNIFIED["vehicle"],    # motorcycle → vehicle
    4:  UNIFIED["vehicle"],    # truck → vehicle
    5:  UNIFIED["vehicle"],    # other-vehicle → vehicle
}

# ---- BotanicGarden (continuous classes) ----
BOTANIC_MAP = {
    0: UNIFIED["unlabeled"],
    1: UNIFIED["ground"],
    2: UNIFIED["vegetation"],
    3: UNIFIED["unlabeled"],   # object → unlabeled
    4: UNIFIED["building"],    # wall → building
}

# ---- S3DIS (indoor, used for wall/floor priors) ----
S3DIS_MAP = {
    "ceiling": UNIFIED["unlabeled"],
    "floor":   UNIFIED["ground"],
    "wall":    UNIFIED["building"],
    "beam":    UNIFIED["building"],
    "column":  UNIFIED["unlabeled"],   # column → unlabeled (no pole)
    "window":  UNIFIED["building"],
    "door":    UNIFIED["building"],
    "table":   UNIFIED["unlabeled"],
    "chair":   UNIFIED["unlabeled"],
    "sofa":    UNIFIED["unlabeled"],
    "bookcase":UNIFIED["unlabeled"],
    "board":   UNIFIED["unlabeled"],
    "clutter": UNIFIED["unlabeled"],
}


def apply_map(labels, mapping):
    import numpy as np
    out = np.zeros_like(labels)
    for k, v in mapping.items():
        out[labels == k] = v
    return out
