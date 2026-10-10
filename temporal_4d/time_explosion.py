"""Time explosion (ARTISTIC): old observations are displaced along a display axis proportional to their age.
Displacement is a visualisation device; true world positions are unchanged and never exported with the offset."""
import numpy as np

AXES = {"up": np.array([0.0, 0.0, 1.0]), "east": np.array([1.0, 0.0, 0.0]), "north": np.array([0.0, 1.0, 0.0])}


def displacement(age_s, metres_per_s, axis="up", clamp_m=600.0):
    a = np.clip(np.asarray(age_s, dtype=np.float64), 0.0, None)
    off = np.minimum(a * metres_per_s, clamp_m)
    return off[..., None] * AXES[axis]
