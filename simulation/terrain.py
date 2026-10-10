"""Simulation terrain: REAL gridded bathymetry (ray-marched) plus clearly flagged SYNTHETIC objects inserted into the
truth surface so detection can be validated against known ground truth."""
import numpy as np
from scipy.ndimage import map_coordinates

from preprocessing.bathymetry_loader import fill_for_raycast
from preprocessing.coordinate_systems import rot_z


def _box_mask(X, Y, cx, cy, length, width, yaw, ramp=1.5):
    c, s = np.cos(yaw), np.sin(yaw)
    u = (X - cx) * c + (Y - cy) * s
    v = -(X - cx) * s + (Y - cy) * c
    a = np.clip((length / 2 - np.abs(u)) / ramp + 0.5, 0, 1)
    b = np.clip((width / 2 - np.abs(v)) / ramp + 0.5, 0, 1)
    return a * b


def choose_object_sites(grid, n, margin, rng, size, min_sep=250.0, max_slope_deg=7.0, bounds=None):
    """Pick flat-ish, valid sites away from the tile edge and from each other."""
    sl = grid.slope_deg()
    ok = np.isfinite(sl) & (sl < max_slope_deg)
    X, Y = np.meshgrid(grid.x, grid.y)
    ok &= (X > grid.x[0] + margin) & (X < grid.x[-1] - margin) & (Y > grid.y[0] + margin) & (Y < grid.y[-1] - margin)
    if bounds is not None:
        ok &= (np.abs(X) < bounds[0]) & (np.abs(Y) < bounds[1])
    cand = np.argwhere(ok)
    rng.shuffle(cand)
    sites = []
    for iy, ix in cand:
        p = np.array([grid.x[ix], grid.y[iy]])
        if all(np.linalg.norm(p - q[:2]) > min_sep for q in sites):
            sites.append(np.array([p[0], p[1], rng.uniform(0, np.pi)]))
        if len(sites) >= n:
            break
    return sites


class SimTerrain:
    def __init__(self, grid, objects=()):
        self.x0, self.y0, self.dx, self.dy = float(grid.x[0]), float(grid.y[0]), grid.dx, grid.dy
        self.valid = np.isfinite(grid.z)
        z = fill_for_raycast(grid.z.copy())
        X, Y = np.meshgrid(grid.x, grid.y)
        self.objects = []
        for o in objects:
            m = _box_mask(X, Y, o["x"], o["y"], o["length"], o["width"], o["yaw"])
            z = z + o["height"] * m
            self.objects.append(dict(o))
        self.z = z
        self.shape = z.shape
        self.extent = (grid.x[0], grid.x[-1], grid.y[0], grid.y[-1])
        gy, gx = np.gradient(z, self.dy, self.dx)
        self._gx, self._gy = gx, gy

    def _coords(self, X, Y):
        return (Y - self.y0) / self.dy, (X - self.x0) / self.dx

    def inside(self, X, Y):
        cy, cx = self._coords(X, Y)
        return (cx >= 0) & (cx <= self.shape[1] - 1) & (cy >= 0) & (cy <= self.shape[0] - 1)

    def height(self, X, Y):
        cy, cx = self._coords(np.asarray(X), np.asarray(Y))
        return map_coordinates(self.z, [cy, cx], order=1, mode="nearest")

    def normal(self, X, Y):
        cy, cx = self._coords(np.asarray(X), np.asarray(Y))
        gx = map_coordinates(self._gx, [cy, cx], order=1, mode="nearest")
        gy = map_coordinates(self._gy, [cy, cx], order=1, mode="nearest")
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def cast(self, origins, dirs, max_range, step=4.0, bisect=12, chunk=4000):
        """March rays from origins (n,3) along unit dirs (n,3) until they pass below the surface.
        Returns r (n,) with NaN = no hit within max_range / left the tile, hit (n,3), normal (n,3)."""
        n = len(origins)
        r_out = np.full(n, np.nan)
        ts = np.arange(step, max_range + step, step)
        for a in range(0, n, chunk):
            o, d = origins[a:a + chunk], dirs[a:a + chunk]
            P = o[:, None, :] + ts[None, :, None] * d[:, None, :]
            h = self.height(P[..., 0], P[..., 1])
            inside = self.inside(P[..., 0], P[..., 1])
            below = (P[..., 2] <= h) & inside
            left = ~inside
            first_below = np.where(below.any(1), below.argmax(1), -1)
            first_left = np.where(left.any(1), left.argmax(1), 10 ** 9)
            hit = (first_below >= 0) & (first_below < first_left)
            lo = np.where(first_below > 0, ts[np.maximum(first_below - 1, 0)], 0.0)
            hi = np.where(first_below >= 0, ts[np.maximum(first_below, 0)], 0.0)
            for _ in range(bisect):
                mid = 0.5 * (lo + hi)
                pm = o + mid[:, None] * d
                b = pm[:, 2] <= self.height(pm[:, 0], pm[:, 1])
                hi = np.where(b, mid, hi)
                lo = np.where(b, lo, mid)
            r = 0.5 * (lo + hi)
            r_out[a:a + chunk] = np.where(hit, r, np.nan)
        ok = np.isfinite(r_out)
        hitp = origins + np.where(ok, r_out, 0.0)[:, None] * dirs
        nrm = self.normal(hitp[:, 0], hitp[:, 1])
        return r_out, hitp, nrm
