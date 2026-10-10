"""Per-point temporal attributes: opacity from age and optional layer colouring. Mirrors the GLSL in
shaders/temporal_points.vert so CPU tests can verify what the GPU shows."""
import numpy as np


def opacity(age_s, decay_s, floor=0.0):
    """Exponential fade: 1 at age 0, `floor` asymptote; age < 0 (not yet observed) -> 0."""
    a = np.asarray(age_s, dtype=np.float64)
    return np.where(a < 0, 0.0, floor + (1.0 - floor) * np.exp(-np.maximum(a, 0.0) / max(decay_s, 1e-6)))


def layer_index(ping_idx, scans_per_layer):
    return (np.asarray(ping_idx) // max(int(scans_per_layer), 1)).astype(np.int64)


def visible_mask(t_point, t_now, history_s):
    """Point is visible if observed at or before t_now and within the history horizon."""
    age = t_now - np.asarray(t_point)
    return (age >= 0) & (age <= history_s)
