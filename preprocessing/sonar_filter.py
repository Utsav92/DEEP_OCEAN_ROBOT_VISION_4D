"""Sounding filters. Each returns a boolean keep-mask; raw arrays are never edited."""
import numpy as np
from scipy.ndimage import median_filter


def invalid_range_mask(ranges, min_range, max_range):
    return np.isfinite(ranges) & (ranges >= min_range) & (ranges <= max_range)


def quality_mask(quality, min_quality):
    return quality >= min_quality


def beam_outlier_mask(depth_below_sonar, valid, window=9, k_sigma=3.0, floor_m=2.0):
    """Reject soundings that deviate from the across-track running median of the same ping.
    depth_below_sonar (N,B); invalid entries should be NaN. Returns keep mask."""
    z = np.where(valid, depth_below_sonar, np.nan)
    fill = np.where(np.isfinite(z), z, np.nanmedian(z) if np.isfinite(z).any() else 0.0)
    med = median_filter(fill, size=(1, window), mode="nearest")
    res = np.abs(fill - med)
    scale = 1.4826 * np.nanmedian(res[valid]) if valid.any() else 1.0
    thr = np.maximum(k_sigma * scale, floor_m)
    return valid & (res <= thr)


def neighbour_ping_outlier_mask(points_world, keep, ping_window=5, k_sigma=3.0, floor_m=3.0):
    """Along-track consistency: compare each sounding height to the median of the same beam over neighbouring pings."""
    z = np.where(keep, points_world[..., 2], np.nan)
    fill = np.where(np.isfinite(z), z, np.nanmedian(z))
    med = median_filter(fill, size=(ping_window, 1), mode="nearest")
    res = np.abs(fill - med)
    scale = 1.4826 * np.nanmedian(res[keep]) if keep.any() else 1.0
    return keep & (res <= np.maximum(k_sigma * scale, floor_m))


def incidence_mask(theta, max_incidence_deg, slope_hint_deg=0.0):
    """Reject outermost beams whose geometric incidence (from nadir) is too grazing for reliable bottom detection."""
    return np.degrees(np.abs(theta)) + slope_hint_deg <= max_incidence_deg


def spike_mask(z_world, valid, window=(5, 9), k_sigma=3.0, floor_m=2.5, max_component=2, hard_reject_m=40.0):
    """Reject ISOLATED spikes only. A sounding is a candidate when it deviates from the 2-D (ping x beam) running median
    by more than max(k_sigma * robust sigma, floor_m); connected groups of candidates with more than `max_component`
    members are kept because they are coherent seabed relief (rocks, wrecks, steps), unless they deviate by more than
    hard_reject_m. Returns keep mask (valid & not spike)."""
    from scipy.ndimage import label
    z = np.where(valid, z_world, np.nan)
    fill = np.where(np.isfinite(z), z, np.nanmedian(z) if np.isfinite(z).any() else 0.0)
    med = median_filter(fill, size=window, mode="nearest")
    res = np.abs(fill - med)
    scale = 1.4826 * np.nanmedian(res[valid]) if valid.any() else 1.0
    cand = valid & (res > max(k_sigma * scale, floor_m))
    lab, n = label(cand, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        big = sizes[lab] > max_component
        coherent = cand & big & (res <= hard_reject_m)
    else:
        coherent = np.zeros_like(cand)
    return valid & (~cand | coherent)
