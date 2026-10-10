"""Bottom detection from amplitude-versus-range profiles (used for side-scan altitude and amplitude-based multibeam
detection QC). Returns quality in 0..1 as peak-to-background contrast."""
import numpy as np


def first_return(profile, threshold_factor=3.0, guard=3):
    """Index of the first sample exceeding `threshold_factor` x the noise level (median of the first quartile)
    after `guard` samples; returns (index, quality) with index -1 if none."""
    p = np.asarray(profile, dtype=np.float64)
    noise = np.median(p[: max(4, len(p) // 4)]) + 1e-9
    above = np.where(p[guard:] > threshold_factor * noise)[0]
    if above.size == 0:
        return -1, 0.0
    i = int(above[0]) + guard
    # refine to the local peak within a few samples
    j = i + int(np.argmax(p[i:i + 5]))
    q = float(np.clip((p[j] / noise - threshold_factor) / (10.0 * threshold_factor), 0.0, 1.0))
    return j, q


def detect_batch(profiles, **kw):
    out = [first_return(p, **kw) for p in profiles]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])
