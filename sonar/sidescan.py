"""Side-scan sonar adapter (optional). Produces a slant-range-corrected backscatter mosaic geometry.

Side-scan gives backscatter vs slant range per side, NOT bathymetry: acoustic intensity is never converted to height
here. Slant-to-ground-range correction uses the flat-bottom assumption with the supplied altitude.
"""
import numpy as np


def slant_to_ground(slant_range, altitude):
    """Ground range (flat-seabed assumption). Returns NaN inside the water-column (range < altitude)."""
    s = np.asarray(slant_range, dtype=np.float64)
    g2 = s ** 2 - altitude ** 2
    return np.where(g2 > 0, np.sqrt(np.maximum(g2, 0.0)), np.nan)


def ground_range_swath(port, starboard, slant_max, altitude, out_bins=512):
    """Resample one ping (port/starboard backscatter vs slant range) to a common ground-range axis.
    Returns (ground_axis_m (signed: port negative), backscatter)."""
    n = len(port)
    s = (np.arange(n) + 0.5) / n * slant_max
    g = slant_to_ground(s, altitude)
    ok = np.isfinite(g)
    gmax = g[ok].max() if ok.any() else 1.0
    axis = np.linspace(-gmax, gmax, out_bins)
    out = np.zeros(out_bins)
    if ok.any():
        sb = np.interp(np.abs(axis), g[ok], np.asarray(starboard)[ok], left=0.0, right=0.0)
        pb = np.interp(np.abs(axis), g[ok], np.asarray(port)[ok], left=0.0, right=0.0)
        out = np.where(axis >= 0, sb, pb)
    return axis, out
