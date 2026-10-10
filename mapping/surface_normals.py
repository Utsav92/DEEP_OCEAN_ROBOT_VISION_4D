"""Surface normals and terrain derivatives from a regular elevation grid."""
import numpy as np


def grid_normals(z, dx, dy):
    """Unit normals (ny, nx, 3) of z(x,y) by central differences; NaN where z or its neighbours are NaN."""
    gy, gx = np.gradient(z, dy, dx)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def slope_deg(z, dx, dy):
    gy, gx = np.gradient(z, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def hillshade(z, dx, dy, azimuth_deg=315.0, altitude_deg=40.0, exaggeration=1.0):
    gy, gx = np.gradient(z * exaggeration, dy, dx)
    sl = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(gy, -gx)
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    return np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp)


def point_normals_pca(points, k=12, tree=None):
    """Per-point normals by local PCA (used for point-to-plane ICP). Returns (normals, curvature, tree)."""
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(points)
    k = min(k, len(points))
    _, nn = tree.query(points, k=k)
    nb = points[nn] - points[nn].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    n = np.where(n[:, 2:3] < 0, -n, n)          # orient upward (seabed normals point to the water column)
    curv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)
    return n, curv, tree
