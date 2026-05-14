CATEGORY_LABELS = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
]

CATEGORY_NAMES = {
    "1a": "bronchial_wall_thickening",
    "1b": "bronchiectasis",
    "1c": "emphysema",
    "1d": "interlobular_septal_thickening",
    "1e": "micronodule",
    "1f": "other_non_focal",
    "2a": "linear_scar",
    "2b": "atelectasis_consolidation",
    "2c": "ground_glass_opacity",
    "2d": "nodule_mass",
    "2e": "pleural_effusion",
    "2f": "honeycombing",
    "2g": "pneumothorax",
    "2h": "other_focal",
}

# Category → expert routing
CATEGORY_TO_EXPERT = {
    "2b": "hu", "2c": "hu", "2e": "hu", "2g": "hu",
    "2d": "nodule_detector",
    "1a": "full_ct_voxtell", "1b": "full_ct_voxtell",
    "1c": "full_ct_voxtell", "2f": "full_ct_voxtell",
}
# Everything not in CATEGORY_TO_EXPERT → "diffuse"

# HU windows per category
HU_THRESHOLDS = {
    "1c": {"lower": None,  "upper": -950},
    "2b": {"lower": -100,  "upper": None},
    "2c": {"lower": -750,  "upper": -300},
    "2e": {"lower": -20,   "upper": 40},
    "2g": {"lower": None,  "upper": -900},
}

# Connected-component connectivity (6 = face-neighbors)
CC_STRUCTURE = 6

# Morphology: min component voxels per category
MIN_COMPONENT_VOXELS = {
    "default": 125,
    "1c": 200,
    "1e": 8,
    "2b": 200,
    "2c": 200,
    "2d": 8,
    "2e": 200,
    "2g": 200,
}
MAX_ELONGATION = 5.0

# Proposal fusion
EXPAND_MARGIN_MM = [10, 10, 10]
UNION_IOU_THRESHOLD = 0.5

# Expert priority (lower = earlier in pipeline)
EXPERT_PRIORITY = {"hu": 0, "nodule_detector": 1, "diffuse": 2}

# MONAI lung_nodule_ct_detection bundle
NODULE_DETECTOR_BUNDLE = "models/MONAI_lung_nodule_ct_detection"
NODULE_DETECTOR_SCORE_THRESHOLDS = {
    "1e": 0.001,
    "2d": 0.001,
}
NODULE_DETECTOR_MAX_DETECTIONS = 50
