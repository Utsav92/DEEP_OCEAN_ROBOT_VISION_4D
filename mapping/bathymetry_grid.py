"""Gridding soundings into a bathymetric raster with coverage, uncertainty and first/last observation time.
Observed cells and interpolated cells are kept in separate arrays so the renderer can distinguish them."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass
class BathyGrid:
    x0: float
    y0: float
    spacing: float
    z: np.ndarray            # (ny,nx) weighted-mean elevation, NaN = not observed
    count: np.ndarray        # (ny,nx) int
    sigma: np.ndarray        # (ny,nx) combined 1-sigma (measurement + along-cell spread), NaN where unobserved
    t_first: np.ndarray      # (ny,nx) s of first observation, NaN where unobserved
    t_last: np.ndarray
    intensity: np.ndarray    # (ny,nx) mean normalised backscatter, NaN where unobserved
    z_interp: np.ndarray     # (ny,nx) z with small gaps filled (NaN beyond max gap)
    interpolated: np.ndarray  # (ny,nx) bool: filled, NOT directly observed

    @property
    def x(self):
        return self.x0 + np.arange(self.z.shape[1]) * self.spacing

    @property
    def y(self):
        return self.y0 + np.arange(self.z.shape[0]) * self.spacing

    @property
    def observed(self):
        return self.count > 0

    def coverage_fraction(self, region_mask=None):
        m = self.observed if region_mask is None else self.observed[region_mask]
        return float(m.mean())


def grid_soundings(xyz, t, intensity, sigma_v, sigma_h, spacing, extent, min_count=1, max_gap_cells=3):
    x0, x1, y0, y1 = extent
    nx = int(np.floor((x1 - x0) / spacing)) + 1
    ny = int(np.floor((y1 - y0) / spacing)) + 1
    ix = np.floor((xyz[:, 0] - x0) / spacing + 0.5).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y0) / spacing + 0.5).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    z = xyz[ok, 2].astype(np.float64)
    sv = np.maximum(sigma_v[ok].astype(np.float64), 0.05)
    w = 1.0 / sv ** 2
    n = np.bincount(flat, minlength=nx * ny)
    sw = np.bincount(flat, weights=w, minlength=nx * ny)
    swz = np.bincount(flat, weights=w * z, minlength=nx * ny)
    swzz = np.bincount(flat, weights=w * z * z, minlength=nx * ny)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = swz / sw
        var_spread = np.maximum(swzz / sw - mean ** 2, 0.0)
        sig = np.sqrt(1.0 / sw + var_spread)
    tf = np.full(nx * ny, np.inf)
    tl = np.full(nx * ny, -np.inf)
    np.minimum.at(tf, flat, t[ok])
    np.maximum.at(tl, flat, t[ok])
    isum = np.bincount(flat, weights=intensity[ok].astype(np.float64), minlength=nx * ny)
    obs = n >= min_count
    shape = (ny, nx)

    def f(a, fill=np.nan):
        return np.where(obs, a, fill).reshape(shape)

    zg = f(mean)
    # fill only small holes, linearly weighted by nearest observed value; never beyond max_gap_cells
    dist, idx = distance_transform_edt(~obs.reshape(shape), return_indices=True)
    near = zg[tuple(idx)]
    fill_ok = (~obs.reshape(shape)) & (dist <= max_gap_cells)
    z_interp = np.where(obs.reshape(shape), zg, np.where(fill_ok, near, np.nan))
    return BathyGrid(x0, y0, spacing, zg, n.reshape(shape).astype(np.int32), f(sig), f(tf), f(tl),
                     f(isum / np.maximum(n, 1)), z_interp, fill_ok)
