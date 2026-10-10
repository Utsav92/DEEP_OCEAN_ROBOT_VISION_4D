"""Exports consumed by TouchDesigner and external tools. All arrays are float32/uint8 .npy (+ JSON metadata) so the
renderer can memory-map them. Heights are ENU elevations (m, up); TouchDesigner axes are applied in the shader/SOPs."""
import json
import os

import numpy as np


def resample_to_texture(x, y, z, size, extent=None):
    """Bilinear-resample z(x,y) (NaN allowed) to a (size, size) texture over `extent`=(x0,x1,y0,y1). Row 0 = y0 (south).
    NaN is preserved; returns (tex float32, valid_mask bool, extent)."""
    from scipy.ndimage import map_coordinates
    x0, x1, y0, y1 = extent or (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    xs = np.linspace(x0, x1, size)
    ys = np.linspace(y0, y1, size)
    cx = (xs - x[0]) / (x[1] - x[0])
    cy = (ys - y[0]) / (y[1] - y[0])
    CX, CY = np.meshgrid(cx, cy)
    valid = np.isfinite(z)
    zf = np.where(valid, z, 0.0)
    h = map_coordinates(zf, [CY, CX], order=1, mode="constant", cval=0.0)
    w = map_coordinates(valid.astype(np.float64), [CY, CX], order=1, mode="constant", cval=0.0)
    ok = w > 0.999
    tex = np.where(ok, h / np.maximum(w, 1e-9), np.nan).astype(np.float32)
    return tex, ok, (x0, x1, y0, y1)


def save_texture(path_base, tex, extent, meta=None):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    np.save(path_base + ".npy", tex.astype(np.float32))
    d = {"extent_xxyy": [float(v) for v in extent], "shape": list(tex.shape),
         "min": float(np.nanmin(tex)) if np.isfinite(tex).any() else None,
         "max": float(np.nanmax(tex)) if np.isfinite(tex).any() else None,
         "valid_fraction": float(np.isfinite(tex).mean()), "axes": "row0=south(y0) col0=west(x0); values ENU elevation m"}
    if meta:
        d.update(meta)
    with open(path_base + ".json", "w") as f:
        json.dump(d, f, indent=1)
    return d
