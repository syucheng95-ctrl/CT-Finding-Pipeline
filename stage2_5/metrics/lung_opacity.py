"""Lung-opacity specific metrics (GGO, consolidation, nodule, etc.).

Categories: 1c, 1d, 1e, 1f, 2a, 2b, 2c, 2d, 2h
"""

import numpy as np
from scipy.spatial import ConvexHull


def solid_component_ratio(ct, mask):
    """Fraction of lesion voxels > -350 HU (solid/soft-tissue density)."""
    vals = ct[mask > 0]
    if len(vals) == 0:
        return 0.0
    return float((vals > -350).sum() / len(vals))


def ggo_ratio(ct, mask):
    """Fraction of lesion voxels in (-750, -350] HU (ground-glass opacity)."""
    vals = ct[mask > 0]
    if len(vals) == 0:
        return 0.0
    ggo = (vals > -750) & (vals <= -350)
    return float(ggo.sum() / len(vals))


def spiculation_index(mask, spacing):
    """convex_hull_volume / original_mask_volume.

    Smooth lesions are close to 1.0; spiculated lesions > 1.3.
    """
    coords = np.argwhere(mask)
    if len(coords) < 4:
        return 1.0
    coords_mm = coords * np.array(spacing)
    try:
        hull = ConvexHull(coords_mm)
        hull_vol = hull.volume
    except Exception:
        return 1.0
    mask_vol = mask.sum() * float(np.prod(spacing))
    if mask_vol < 1e-9:
        return 1.0
    return float(hull_vol / mask_vol)


def pleural_distance_mm(mask, spacing):
    """Shortest distance from lesion surface to the ROI boundary.

    Proxy for pleural proximity: smaller = closer to pleura / chest wall.
    A true pleural-distance would need lung segmentation; this is an
    upper-bound approximation using the cropped ROI extent.
    """
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0.0
    shape = np.array(mask.shape)
    # Distance to each of the 6 ROI faces
    dists = []
    for axis in range(3):
        d_lo = coords[:, axis].min()
        d_hi = shape[axis] - 1 - coords[:, axis].max()
        dists.extend([d_lo * spacing[axis], d_hi * spacing[axis]])
    return float(min(dists))


def cavity_fraction(ct, mask):
    """Fraction of lesion voxels < -900 HU (air / necrosis)."""
    vals = ct[mask > 0]
    if len(vals) == 0:
        return 0.0
    return float((vals < -900).sum() / len(vals))


def compute_lung_opacity_metrics(ct, mask, spacing):
    return {
        "solid_component_ratio": round(solid_component_ratio(ct, mask), 3),
        "ggo_ratio": round(ggo_ratio(ct, mask), 3),
        "spiculation_index": round(spiculation_index(mask, spacing), 3),
        "pleural_distance_mm": round(pleural_distance_mm(mask, spacing), 2),
        "cavity_fraction": round(cavity_fraction(ct, mask), 3),
    }
