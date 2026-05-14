"""Route lesion to metric group and compute all applicable measurements."""

from .basic import compute_basic_metrics
from .lung_opacity import compute_lung_opacity_metrics
from .airway_change import compute_airway_change_metrics
from .pleural import compute_pleural_metrics

# 14-category → 3 metric-group mapping (from README / classifier config)
CATEGORY_TO_GROUP = {
    "1a": "airway_change", "1b": "airway_change",
    "1c": "lung_opacity", "1d": "lung_opacity", "1e": "lung_opacity",
    "1f": "lung_opacity", "2a": "lung_opacity", "2b": "lung_opacity",
    "2c": "lung_opacity", "2d": "lung_opacity", "2h": "lung_opacity",
    "2e": "pleural", "2f": "pleural", "2g": "pleural",
}


def compute_all_metrics(ct, mask, spacing, category):
    """Compute all applicable metrics for a single lesion.

    Args:
        ct: np.ndarray (H, W, D)  CT intensities in HU.
        mask: np.ndarray (H, W, D) binary foreground mask (0/1).
        spacing: tuple (sx, sy, sz) voxel spacing in mm.
        category: str, one of the 14 fine-grained lesion categories
                  (e.g. "2c").

    Returns:
        dict with keys "metric_group", "basic", and the group-specific
        sub-dict ("lung_opacity" / "airway_change" / "pleural").
    """
    group = CATEGORY_TO_GROUP.get(category, "lung_opacity")

    result = {
        "metric_group": group,
        "basic": compute_basic_metrics(ct, mask, spacing),
    }

    if group == "lung_opacity":
        result["lung_opacity"] = compute_lung_opacity_metrics(ct, mask, spacing)
    elif group == "airway_change":
        result["airway_change"] = compute_airway_change_metrics(ct, mask, spacing)
    elif group == "pleural":
        result["pleural"] = compute_pleural_metrics(ct, mask, spacing)

    return result
