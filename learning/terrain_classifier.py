"""Unsupervised terrain REGIMES: k-means on auto-encoder latents (or hand features). Cluster ids are NOT semantic labels
(no annotations exist to justify 'rock' or 'sand'); they only group patches of similar relief."""
import numpy as np


def kmeans(x, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        nc = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return a, c


def hand_features(patches):
    """Slope magnitude, roughness (std), curvature proxy per patch (m units preserved)."""
    gy, gx = np.gradient(patches, axis=(1, 2))
    slope = np.hypot(gx, gy).mean(axis=(1, 2))
    rough = patches.std(axis=(1, 2))
    lap = np.abs(np.gradient(gx, axis=2) + np.gradient(gy, axis=1)).mean(axis=(1, 2))
    return np.column_stack([slope, rough, lap])


def silhouette(x, labels, max_n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(len(x), max_n), replace=False)
    x, l = x[idx], labels[idx]
    D = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    s = []
    for i in range(len(x)):
        same = (l == l[i])
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, l == j].mean() for j in np.unique(l) if j != l[i]) if len(np.unique(l)) > 1 else a
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))
