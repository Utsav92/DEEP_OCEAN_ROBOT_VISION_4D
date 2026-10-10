"""Acoustic intensity processing: normalisation and speckle reduction (raw data are never modified in place)."""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def normalise(intensity, valid=None, percentile=(2.0, 98.0)):
    """Robust normalisation to 0..1 using percentiles of valid samples."""
    a = np.asarray(intensity, dtype=np.float64)
    v = a[np.isfinite(a) & (valid if valid is not None else True)]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, percentile)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def apply_gain(intensity, gain=1.0, gamma=1.0):
    return np.clip((np.asarray(intensity) * gain), 0.0, 1.0) ** gamma


def speckle_median(img, window=3):
    return median_filter(np.asarray(img, dtype=np.float64), size=window, mode="nearest")


def speckle_lee(img, window=5, enl=4.0):
    """Lee speckle filter for multiplicative noise (ENL = equivalent number of looks)."""
    a = np.asarray(img, dtype=np.float64)
    mean = uniform_filter(a, window)
    var = np.maximum(uniform_filter(a * a, window) - mean ** 2, 0.0)
    cu2 = 1.0 / enl
    ci2 = var / np.maximum(mean ** 2, 1e-12)
    w = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-12), 0.0, 1.0)
    return mean + w * (a - mean)
