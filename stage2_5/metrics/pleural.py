"""Pleural-lesion specific metrics.

Categories: 2e (pleural effusion), 2f (pleural fibrosis), 2g (pneumothorax).

Pneumothorax ratio and pleural thickness require full-lung context and are
approximated here from the ROI alone.  Effusion volume works directly.
"""

import numpy as np


def effusion_volume_ml(mask, spacing):
    """Pleural effusion volume (same as basic volume, named for clarity)."""
    from .basic import volume_ml
    return volume_ml(mask, spacing)


def pleural_thickness_mm(mask, spacing):
    """Estimate pleural thickening by projecting mask onto its shortest axis.

    For fibrosis (2f): the mask is a thin plaque along the pleura.  We take
    the minimum extent among the 3 PCA axes as an estimate of thickness.
    """
    coords = np.argwhere(mask).astype(np.float64)
    if len(coords) < 3:
        return 0.0
    coords_mm = coords * np.array(spacing)
    centroid = coords_mm.mean(axis=0)
    cov = np.cov((coords_mm - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    proj_min = np.array([np.dot(coords_mm - centroid, eigvecs[:, i])
                          for i in range(3)]).min(axis=1).min()
    return float(abs(proj_min) * 2)  # approximate thickness


def pneumothorax_fraction(ct, mask, spacing):
    """Fraction of ROI that is air (< -900 HU) within the lesion mask.

    For pneumothorax (2g): the mask should capture the air-filled pleural
    space.  Note: accurate measurement requires comparison with the
    ipsilateral thoracic cavity, which is not available from the ROI alone.
    """
    if mask.sum() < 1:
        return 0.0
    air = (ct[mask > 0] < -900).sum()
    return float(air / mask.sum())


def compute_pleural_metrics(ct, mask, spacing):
    return {
        "effusion_volume_ml": round(effusion_volume_ml(mask, spacing), 4),
        "pleural_thickness_mm": round(pleural_thickness_mm(mask, spacing), 2),
        "pneumothorax_fraction": round(pneumothorax_fraction(ct, mask, spacing), 3),
    }
