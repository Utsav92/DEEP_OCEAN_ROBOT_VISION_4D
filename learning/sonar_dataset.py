"""Self-supervised patch dataset from the gridded bathymetry (no labels exist, so none are invented).
Each patch is a PxP window of the reconstructed seabed with its mean removed (relief only)."""
import numpy as np


def extract_patches(z, observed, size=32, stride=16, min_valid=0.95):
    ys, xs, out = [], [], []
    ny, nx = z.shape
    for iy in range(0, ny - size + 1, stride):
        for ix in range(0, nx - size + 1, stride):
            m = observed[iy:iy + size, ix:ix + size]
            if m.mean() < min_valid:
                continue
            p = z[iy:iy + size, ix:ix + size].astype(np.float32)
            p = p - p.mean()
            out.append(p)
            ys.append(iy)
            xs.append(ix)
    if not out:
        return np.zeros((0, size, size), np.float32), np.zeros((0, 2), int)
    return np.stack(out), np.column_stack([ys, xs])


def normalise(patches, scale=None):
    s = float(np.percentile(np.abs(patches), 99)) if scale is None else scale
    return np.clip(patches / max(s, 1e-6), -1.5, 1.5), s


def split(n, frac=0.8, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]
