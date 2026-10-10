"""Acoustic features and keyframe submaps for SLAM.

Multibeam: keyframe submaps (point clouds over +-window metres of dead-reckoned path), rasterised bathymetry patches
for correlation, and curvature-based saliency. Imaging sonar (FLS): bright-return feature points in Cartesian sonar
coordinates (CFAR-style threshold) - sonar specific, NOT RGB feature matching.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter

from preprocessing.coordinate_systems import invert


@dataclass
class Keyframes:
    ping: np.ndarray                 # (K,) ping index of each keyframe
    t: np.ndarray                    # (K,)
    s: np.ndarray                    # (K,) dead-reckoned path length
    T0: np.ndarray                   # (K,4,4) dead-reckoned T_world_vehicle
    clouds: list                     # K arrays (n,3) in the keyframe vehicle frame
    sigma_xy: np.ndarray             # (K,) dead-reckoning horizontal 1-sigma
    window: tuple = (0, 0)
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)

    def world_cloud(self, k, T=None):
        T = self.T0[k] if T is None else T
        c = self.clouds[k]
        return c @ T[:3, :3].T + T[:3, 3]


def build_keyframes(snd, ping_s, ping_sigma, spacing_m=30.0, window_m=60.0, max_points=2500, seed=0):
    """snd: dict of arrays from soundings_dr.npz (ping, xyz, T_world_vehicle, ping_t); ping_s: (N,) DR path length per
    ping; ping_sigma: (N,) DR sigma. Keyframes every `spacing_m` of path with a submap of pings within +-window_m."""
    rng = np.random.default_rng(seed)
    N = len(ping_s)
    marks = np.arange(0.0, ping_s[-1], spacing_m)
    kp = np.unique(np.clip(np.searchsorted(ping_s, marks), 0, N - 1))
    ping = snd["ping"]
    first = np.searchsorted(ping, np.arange(N + 1))            # sounding offsets per ping (ping sorted ascending)
    Tw = snd["T_world_vehicle"]
    clouds, keep_kp = [], []
    for k in kp:
        lo = np.searchsorted(ping_s, ping_s[k] - window_m)
        hi = np.searchsorted(ping_s, ping_s[k] + window_m, side="right")
        pts = snd["xyz"][first[lo]:first[hi]].astype(np.float64)
        if len(pts) < 200:
            continue
        Ti = invert(Tw[k])
        loc = pts @ Ti[:3, :3].T + Ti[:3, 3]
        if len(loc) > max_points:
            loc = loc[rng.choice(len(loc), max_points, replace=False)]
        clouds.append(loc.astype(np.float32))
        keep_kp.append(k)
    kp = np.array(keep_kp)
    return Keyframes(ping=kp, t=snd["ping_t"][kp], s=ping_s[kp], T0=Tw[kp], clouds=clouds, sigma_xy=ping_sigma[kp],
                     window=(spacing_m, window_m))


def saliency(points, k=10):
    """Local roughness (std of z over k neighbours) - flat featureless seabed has low saliency and registers poorly."""
    from scipy.spatial import cKDTree
    t = cKDTree(points[:, :2])
    _, nn = t.query(points[:, :2], k=min(k, len(points)))
    return points[nn, 2].std(axis=1)


def raster_patch(points, cell, margin=0.0):
    """Rasterise world-frame points on a grid anchored at integer multiples of `cell`. Returns (z, mask, x0, y0)."""
    x0 = np.floor(points[:, 0].min() / cell) * cell - margin
    y0 = np.floor(points[:, 1].min() / cell) * cell - margin
    nx = int(np.ceil((points[:, 0].max() - x0) / cell)) + 1 + int(2 * margin / cell)
    ny = int(np.ceil((points[:, 1].max() - y0) / cell)) + 1 + int(2 * margin / cell)
    return x0, y0, (ny, nx)


def fls_features(img01, max_range, bearings, thr_sigma=3.0, win=9, min_val=0.25):
    """CFAR-like bright feature extraction in a polar FLS image (n_range, n_bearing): pixel is a feature if it exceeds
    the local mean by thr_sigma local standard deviations. Returns (n,2) points (forward x, starboard y) in metres."""
    a = np.asarray(img01, dtype=np.float64)
    mu = uniform_filter(a, win)
    sd = np.sqrt(np.maximum(uniform_filter(a * a, win) - mu * mu, 1e-9))
    hit = (a > mu + thr_sigma * sd) & (a > min_val)
    ir, ib = np.nonzero(hit)
    r = (ir + 0.5) / a.shape[0] * max_range
    b = bearings[0] + ib / (a.shape[1] - 1) * (bearings[-1] - bearings[0])
    return np.column_stack([r * np.cos(b), r * np.sin(b)])
