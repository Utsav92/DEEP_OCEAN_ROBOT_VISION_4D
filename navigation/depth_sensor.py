"""Pressure / depth sensor model and conversion (UNESCO 1983 depth-from-pressure, Saunders 1981)."""
import numpy as np


def gravity(lat_deg, P_dbar):
    x = np.sin(np.radians(lat_deg)) ** 2
    return 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * P_dbar


def depth_from_pressure(P_dbar, lat_deg):
    """Depth in metres from pressure in decibar (UNESCO/Saunders). Accurate to ~1 m at 10000 dbar."""
    P = np.asarray(P_dbar, dtype=np.float64)
    num = (((-1.82e-15 * P + 2.279e-10) * P - 2.2512e-5) * P + 9.72659) * P
    return num / gravity(lat_deg, P)


def pressure_from_depth(depth_m, lat_deg, iters=6):
    """Inverse of depth_from_pressure by fixed-point iteration."""
    d = np.asarray(depth_m, dtype=np.float64)
    P = d.copy()
    for _ in range(iters):
        est = depth_from_pressure(P, lat_deg)
        P = P + (d - est) / 1.0 * 1.0
    return P


class DepthSensor:
    """Zero-mean noise plus a slowly varying bias, returned as depth (m, positive down)."""

    def __init__(self, noise_m=0.08, bias_m=0.0, lat_deg=36.77):
        self.noise, self.bias, self.lat = noise_m, bias_m, lat_deg

    def measure(self, depth_true, rng):
        d = np.asarray(depth_true, dtype=np.float64)
        P = pressure_from_depth(d, self.lat)
        P_meas = P + rng.normal(0.0, self.noise / 1.0, d.shape) * 1.0 + self.bias   # ~1 dbar per metre
        return depth_from_pressure(P_meas, self.lat)
