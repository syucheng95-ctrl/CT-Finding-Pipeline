"""Airway-change specific metrics.

Categories: 1a (bronchial wall thickening), 1b (bronchiectasis / dilation).

The full pipeline would skeletonize the airway mask and measure wall/lumen
diameters.  Here we provide a simplified version based on mask morphology.
"""

import numpy as np
from scipy.ndimage import binary_erosion


def _inner_lumen(mask, spacing):
    """Estimate the lumen region by eroding the binary mask.

    Erosion radius ~ 1 mm removes the wall, leaving the air-filled lumen.
    """
    # 1 mm erosion radius in voxels (per axis)
    radius = tuple(max(1, int(round(1.0 / s))) for s in spacing)
    if mask.sum() < 5:
        return mask
    # Use scipy binary_erosion with approximate structuring element
    # Simple: erode in 3D with 1-voxel radius, iterate
    lumen = mask.copy()
    for _ in range(max(radius)):
        lumen = binary_erosion(lumen)
        if lumen.sum() < 1:
            break
    return lumen.astype(np.uint8)


def wall_thickness_ratio(mask, spacing):
    """Estimated ratio: wall volume / (wall + lumen) volume.

    Higher values suggest thicker walls.
    """
    total = mask.sum()
    if total < 5:
        return 0.0
    lumen = _inner_lumen(mask, spacing)
    lumen_vox = lumen.sum()
    wall_vox = max(0, total - lumen_vox)
    return float(wall_vox / total)


def lumen_dilation_index(mask, spacing):
    """Ratio: estimated lumen diameter / estimated outer diameter.

    Values > 0.7 suggest bronchiectasis (dilated airway).
    """
    from .basic import effective_diameter_mm

    outer_diam = effective_diameter_mm(mask, spacing)
    lumen = _inner_lumen(mask, spacing)
    if lumen.sum() < 3:
        return 0.0
    inner_diam = effective_diameter_mm(lumen, spacing)
    if outer_diam < 0.01:
        return 0.0
    return float(inner_diam / outer_diam)


def compute_airway_change_metrics(ct, mask, spacing):
    return {
        "wall_thickness_ratio": round(wall_thickness_ratio(mask, spacing), 3),
        "lumen_dilation_index": round(lumen_dilation_index(mask, spacing), 3),
    }
