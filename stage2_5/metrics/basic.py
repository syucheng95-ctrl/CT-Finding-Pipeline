"""Generic metrics computed for all lesion types."""

import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes, mesh_surface_area


def _voxel_volume_ml(spacing):
    return float(np.prod(spacing)) / 1000.0


def volume_ml(mask, spacing):
    """Lesion volume in milliliters."""
    vox_mm3 = float(np.prod(spacing))
    return mask.sum() * vox_mm3 / 1000.0


def max_diameter_mm(mask, spacing):
    """PCA major axis span through the lesion, in mm.

    Uses weighted PCA on mask voxel coordinates: the longest principal-axis
    extent of the point cloud approximates the RECIST-like maximum diameter.
    """
    coords = np.argwhere(mask).astype(np.float64)  # (N, 3) in voxel units
    if len(coords) < 3:
        return 0.0
    # Weight by spacing before PCA so that distances reflect mm
    coords_mm = coords * np.array(spacing)
    centroid = coords_mm.mean(axis=0)
    cov = np.cov((coords_mm - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Project points onto dominant eigenvector, measure span
    proj = (coords_mm - centroid) @ eigvecs[:, -1]
    return float(proj.max() - proj.min())


def effective_diameter_mm(mask, spacing):
    """Diameter of a sphere with equal volume, in mm."""
    vol_mm3 = mask.sum() * float(np.prod(spacing))
    return float(2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))


def mean_hu(ct, mask):
    """Mean CT attenuation (HU) inside the lesion."""
    vals = ct[mask > 0]
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def sphericity(mask, spacing):
    """Ratio: surface area of volume-equivalent sphere / actual surface area.

    Values closer to 1.0 = more spherical.  Calculated from marching-cubes
    mesh so requires `pip install scikit-image`.
    """
    if mask.sum() < 10:
        return 1.0
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5,
                                             spacing=spacing)
        surf_area = mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        return 1.0

    vol_mm3 = mask.sum() * float(np.prod(spacing))
    # Radius of equal-volume sphere
    r = (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    sphere_area = 4.0 * np.pi * r * r
    if surf_area < 1e-9:
        return 1.0
    return float(sphere_area / surf_area)


def compute_basic_metrics(ct, mask, spacing):
    return {
        "volume_ml": round(volume_ml(mask, spacing), 4),
        "max_diameter_mm": round(max_diameter_mm(mask, spacing), 2),
        "effective_diameter_mm": round(effective_diameter_mm(mask, spacing), 2),
        "mean_hu": round(mean_hu(ct, mask), 1),
        "sphericity": round(sphericity(mask, spacing), 3),
    }
