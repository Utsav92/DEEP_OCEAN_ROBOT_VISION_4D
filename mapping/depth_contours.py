"""Depth contour extraction (marching squares via scikit-image) in local ENU metres."""
import numpy as np


def contours(grid_x, grid_y, z, interval=25.0, levels=None):
    """Returns a list of dict(level, polyline Nx2 (x,y) ) for elevation iso-lines. NaN cells break lines."""
    from skimage import measure
    zf = np.where(np.isfinite(z), z, np.nan)
    if levels is None:
        lo, hi = np.nanmin(zf), np.nanmax(zf)
        levels = np.arange(np.ceil(lo / interval) * interval, hi, interval)
    out = []
    dx, dy = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    zz = np.where(np.isfinite(zf), zf, -1e9)
    for lv in levels:
        for c in measure.find_contours(zz, lv):
            if len(c) < 4:
                continue
            xy = np.stack([grid_x[0] + c[:, 1] * dx, grid_y[0] + c[:, 0] * dy], -1)
            out.append({"level": float(lv), "xy": xy})
    return out


def contour_segments_enu(cs, grid=None, z_lift=0.0):
    """Flatten contour polylines to an (N,2,3) array of line segments at their own elevation (ENU)."""
    segs = []
    for c in cs:
        p = c["xy"]
        a, b = p[:-1], p[1:]
        z = np.full(len(a), c["level"] + z_lift)
        segs.append(np.stack([np.column_stack([a, z]), np.column_stack([b, z])], 1))
    return np.concatenate(segs, 0) if segs else np.zeros((0, 2, 3))
