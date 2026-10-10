"""Doppler velocity log: bottom-track body velocity model and dead-reckoning integration with dropout handling."""
import numpy as np

from preprocessing.coordinate_systems import R_enu_body


class DvlModel:
    def __init__(self, scale_error=0.0015, noise_mps=0.012, bias_mps=(0.004, -0.003, 0.0), dropouts=()):
        self.k, self.noise, self.bias = scale_error, noise_mps, np.asarray(bias_mps, dtype=np.float64)
        self.dropouts = [tuple(d) for d in dropouts]

    def valid_mask(self, t):
        ok = np.ones(len(t), dtype=bool)
        for a, b in self.dropouts:
            ok &= ~((t >= a) & (t <= b))
        return ok

    def measure(self, t, v_body_true, rng):
        v = (1.0 + self.k) * np.asarray(v_body_true) + self.bias + rng.normal(0.0, self.noise, v_body_true.shape)
        valid = self.valid_mask(t)
        v = np.where(valid[:, None], v, np.nan)
        return v, valid


def dead_reckon(t, roll, pitch, yaw, v_body, valid, depth, p0_xy=(0.0, 0.0)):
    """Integrate body velocity rotated by the measured attitude. Bottom-lock gaps hold the last valid velocity
    (constant-velocity extrapolation) and are reported in the returned `extrapolated` mask. z is the depth-sensor depth.
    Returns positions ENU (N,3), extrapolated mask (N,)."""
    n = len(t)
    pos = np.zeros((n, 3))
    pos[0, :2] = p0_xy
    extrap = ~valid
    v_last = np.array([0.0, 0.0, 0.0])
    vh = np.zeros((n, 3))
    for i in range(n):
        if valid[i] and np.all(np.isfinite(v_body[i])):
            v_last = v_body[i]
        vh[i] = v_last
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        R0, R1 = R_enu_body(roll[i - 1], pitch[i - 1], yaw[i - 1]), R_enu_body(roll[i], pitch[i], yaw[i])
        v_enu = 0.5 * (R0 @ vh[i - 1] + R1 @ vh[i])
        pos[i, :2] = pos[i - 1, :2] + v_enu[:2] * dt
    pos[:, 2] = -np.asarray(depth)
    return pos, extrap
