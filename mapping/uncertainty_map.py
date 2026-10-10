"""Mapping uncertainty from the gridded soundings: combined per-cell 1-sigma (sensor model + inter-pass spread) and a
bounded 0..1 display value. No uncertainty is invented for unobserved cells (they stay NaN)."""
import numpy as np


def display_value(sigma, lo=0.2, hi=3.0):
    s = np.asarray(sigma, dtype=np.float64)
    return np.where(np.isfinite(s), np.clip((s - lo) / (hi - lo), 0.0, 1.0), np.nan)


def summary(grid):
    s = grid.sigma[grid.observed]
    return {"cells_observed": int(grid.observed.sum()), "sigma_median_m": float(np.median(s)),
            "sigma_p90_m": float(np.percentile(s, 90)), "multi_pass_cells_fraction": float(np.mean(grid.count[grid.observed] > 60))}
