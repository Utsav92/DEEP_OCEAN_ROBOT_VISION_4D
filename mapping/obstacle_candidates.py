"""Acoustic obstacle candidates from positive seabed relief (geometric, no semantic labels).

Method: morphological opening of the observed bathymetric grid estimates the local background surface; the top-hat
residual (z - opening) marks relief that is narrower than the structuring element. Connected regions above a height
threshold with enough area are candidates. Classification status is always 'unclassified': no annotations or validated
model exist to justify a semantic label.
"""
import numpy as np
from scipy import ndimage as ndi


def detect(grid, opening_m=36.0, min_height_m=3.5, min_area_m2=60.0, source="multibeam", max_extent_m=60.0,
           max_area_m2=2500.0):
    z = grid.z_interp.copy()
    obs = grid.observed
    fill = np.where(np.isfinite(z), z, np.nanmin(z))
    k = max(3, int(round(opening_m / grid.spacing)) | 1)
    bg = ndi.grey_opening(fill, size=(k, k))
    top = fill - bg
    top[~obs] = 0.0                                        # only directly observed cells can raise a candidate
    mask = top >= min_height_m
    lab, n = ndi.label(mask)
    cands = []
    cell_a = grid.spacing ** 2
    for k_ in range(1, n + 1):
        m = lab == k_
        area = float(m.sum() * cell_a)
        if area < min_area_m2 or area > max_area_m2:
            continue
        iy, ix = np.nonzero(m)
        if max((ix.max() - ix.min() + 1), (iy.max() - iy.min() + 1)) * grid.spacing > max_extent_m:
            continue                                           # canyon walls / ridges are not compact obstacles
        h = float(top[m].max())
        sig = float(np.nanmean(grid.sigma[m])) if np.isfinite(grid.sigma[m]).any() else float("nan")
        cx = float(grid.x0 + ix.mean() * grid.spacing)
        cy = float(grid.y0 + iy.mean() * grid.spacing)
        ext = [float((ix.max() - ix.min() + 1) * grid.spacing), float((iy.max() - iy.min() + 1) * grid.spacing)]
        npts = int(grid.count[m].sum())
        snr = h / max(sig, 0.05) if np.isfinite(sig) else 0.0
        conf = float(1.0 / (1.0 + np.exp(-(0.35 * snr - 2.0)))) * min(1.0, npts / 40.0)
        tt = grid.t_first[m]
        cands.append({"id": len(cands), "timestamp_s": float(np.nanmedian(tt)), "position_enu_m": [cx, cy, float(np.nanmax(z[m]))],
                      "height_above_background_m": h, "extent_m": ext, "area_m2": area, "n_soundings": npts,
                      "source_sensor": source, "confidence": conf, "classification_status": "unclassified"})
    return cands, top


def match_to_truth(cands, truth_objects, radius_m=15.0):
    """Evaluation against INSERTED synthetic objects: returns precision/recall style counts."""
    used, tp = set(), 0
    for o in truth_objects:
        best, bd = None, 1e9
        for c in cands:
            d = np.hypot(c["position_enu_m"][0] - o["x"], c["position_enu_m"][1] - o["y"])
            if d < bd and c["id"] not in used:
                best, bd = c, d
        if best is not None and bd <= radius_m:
            tp += 1
            used.add(best["id"])
    return {"truth_objects": len(truth_objects), "detected": tp, "candidates": len(cands),
            "recall": tp / max(len(truth_objects), 1),
            "natural_relief_candidates": len(cands) - tp}
