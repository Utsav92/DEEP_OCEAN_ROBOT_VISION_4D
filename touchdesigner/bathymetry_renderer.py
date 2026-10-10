"""Terrain height textures for the bathymetry shader.

kinds: 'survey'   real NOAA gridded bathymetry of the survey tile (3.5 m grid resampled to 512^2)
       'overview' real NOAA gridded bathymetry of the wider region (17.5 m grid resampled to 256^2)
       'recon'    seabed reconstructed from the (simulated) multibeam: r = elevation or -1e4 if unobserved,
                  g = first-observation time s, b = sigma m, a = sounding count
All elevations are metres (ENU up); the shader subtracts z_ref and applies the visual exaggeration.
"""
import json
import os

import numpy as np

BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"


def _terrain_dir(base):
    return os.path.join(os.path.dirname(base.fetch("data_dir")), "terrain")


def extent(base, kind):
    if kind == "recon":
        st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
        return tuple(st.meta["recon_extent_xxyy"])
    meta = json.load(open(os.path.join(_terrain_dir(base), "terrain_%s.json" % kind)))
    return tuple(meta["extent_xxyy"])


def array(base, kind):
    if kind == "recon":
        st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
        return np.ascontiguousarray(np.asarray(st.array("recon_tex"), dtype=np.float32))
    z = np.load(os.path.join(_terrain_dir(base), "terrain_%s.npy" % kind)).astype(np.float32)
    z = np.where(np.isfinite(z), z, np.float32(np.nanmedian(z)))
    out = np.zeros(z.shape + (4,), np.float32)
    out[..., 0] = z
    out[..., 3] = 1.0
    return out


def cook(scriptOp, kind):
    import td
    arr = array(td.op(BASE), kind)
    scriptOp.par.outputresolution = "custom"
    scriptOp.par.resolutionw = arr.shape[1]
    scriptOp.par.resolutionh = arr.shape[0]
    scriptOp.copyNumpyArray(arr)
    return arr.shape


def scene_info(base):
    """Scene numbers used by cameras/UI: centre and extents in TD display axes (y = elevation - z_ref)."""
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    zr = float(st.meta["z_ref_m"])
    x0, x1, y0, y1 = extent(base, "survey")
    arr = np.load(os.path.join(_terrain_dir(base), "terrain_survey.npy"))
    lo, hi = float(np.nanmin(arr)) - zr, float(np.nanmax(arr)) - zr
    return {"extent_xxzz": [x0, x1, -y1, -y0], "center": [0.5 * (x0 + x1), 0.5 * (lo + hi), -0.5 * (y0 + y1)],
            "floor_y": lo, "top_y": hi, "z_ref": zr, "elev_min": lo + zr, "elev_max": hi + zr}
