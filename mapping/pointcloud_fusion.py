"""Point-cloud fusion helpers: voxel downsampling, display subset selection (stable, time-ordered) and tiling."""
import numpy as np


def voxel_downsample(points, voxel, weights=None):
    """Mean point per voxel. Returns (centres (m,3), counts (m,), first_index_per_voxel (m,))."""
    q = np.floor(points / voxel).astype(np.int64)
    q -= q.min(axis=0)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    first = np.full(len(uniq), len(points), dtype=np.int64)
    np.minimum.at(first, inv, np.arange(len(points)))
    out = np.stack([np.bincount(inv, weights=points[:, k]) / cnt for k in range(3)], axis=1)
    return out, cnt, first


def select_display_subset(n_total, max_points, seed=3):
    """Deterministic uniform subset of indices, returned in ascending (= acquisition-time) order."""
    if n_total <= max_points:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, max_points, replace=False))


def tile_index(points, tile=250.0):
    """Spatial tile id per point and a dict tile -> index array (for culling / LOD)."""
    ij = np.floor(points[:, :2] / tile).astype(np.int64)
    key = ij[:, 0] * 100003 + ij[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    return key, {int(ks[a]): order[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ks)])}
