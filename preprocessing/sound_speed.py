"""Sound speed in sea water and range conversion.

Mackenzie (1981) nine-term equation, valid T 2-30 C, S 25-40 psu, D 0-8000 m (accuracy ~0.07 m/s).
Two-way travel time to range:  r = c * dt / 2   (only for travel-time data; calibrated ranges are NOT converted again).
"""
import numpy as np


def mackenzie(T, S, D):
    """Sound speed (m/s) from temperature (C), salinity (psu), depth (m)."""
    T, S, D = np.asarray(T, dtype=np.float64), np.asarray(S, dtype=np.float64), np.asarray(D, dtype=np.float64)
    return (1448.96 + 4.591 * T - 5.304e-2 * T ** 2 + 2.374e-4 * T ** 3 + 1.340 * (S - 35.0)
            + 1.630e-2 * D + 1.675e-7 * D ** 2 - 1.025e-2 * T * (S - 35.0) - 7.139e-13 * T * D ** 3)


class SoundSpeedProfile:
    """Depth (positive down, m) vs sound speed (m/s), linearly interpolated."""

    def __init__(self, depth, c):
        d = np.asarray(depth, dtype=np.float64)
        order = np.argsort(d)
        self.depth, self.c = d[order], np.asarray(c, dtype=np.float64)[order]

    @classmethod
    def from_tsd(cls, rows, bias_mps=0.0):
        """rows = [(depth, temperature_C, salinity_psu), ...]"""
        r = np.asarray(rows, dtype=np.float64)
        return cls(r[:, 0], mackenzie(r[:, 1], r[:, 2], r[:, 0]) + bias_mps)

    def at(self, depth):
        return np.interp(depth, self.depth, self.c)

    def mean_between(self, d0, d1, n=64):
        """Harmonic mean sound speed (the travel-time-correct average for a vertical path) between two depths."""
        d = np.linspace(d0, d1, n)
        return float(1.0 / np.mean(1.0 / self.at(d)))


def _slowness_integral(self):
    if not hasattr(self, "_zg"):
        self._zg = np.linspace(min(0.0, self.depth[0]), max(self.depth[-1], 12000.0), 24001)
        s = 1.0 / self.at(self._zg)
        self._I = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(self._zg))])


def harmonic_mean(self, z1, z2):
    """Vectorised travel-time-correct mean sound speed between depths z1 and z2 (any shapes, broadcast)."""
    _slowness_integral(self)
    z1, z2 = np.broadcast_arrays(np.asarray(z1, dtype=np.float64), np.asarray(z2, dtype=np.float64))
    I1, I2 = np.interp(z1, self._zg, self._I), np.interp(z2, self._zg, self._I)
    dz = z2 - z1
    with np.errstate(divide="ignore", invalid="ignore"):
        c = dz / (I2 - I1)
    return np.where(np.abs(dz) < 1e-6, self.at(z1), c)


SoundSpeedProfile.harmonic_mean = harmonic_mean


def range_from_twtt_profile(twtt, theta, depth_sonar, profile, iters=4):
    """Slant range from two-way travel time through a sound-speed profile (straight-ray, depth-averaged c).
    theta = across-track angle from nadir (rad). Converges in a few fixed-point iterations."""
    twtt = np.asarray(twtt, dtype=np.float64)
    r = profile.at(depth_sonar) * twtt / 2.0
    for _ in range(iters):
        c = profile.harmonic_mean(depth_sonar, depth_sonar + r * np.cos(theta))
        r = c * twtt / 2.0
    return r


def range_from_twtt(twtt, c):
    """r = c * dt / 2"""
    return np.asarray(c) * np.asarray(twtt, dtype=np.float64) / 2.0


def twtt_from_range(r, c):
    return 2.0 * np.asarray(r, dtype=np.float64) / np.asarray(c)


def ray_trace(theta0, twtt, profile, z_start_depth, n_steps=400):
    """Refracted ray tracing through the profile by Snell's law (p = sin(theta)/c constant).

    theta0: launch angle from the VERTICAL (rad); twtt: two-way travel time (s); z_start_depth: transducer depth (m).
    Returns (horizontal_offset m, vertical_offset m positive down, slant_chord m). For a constant profile this reduces
    to the straight-ray result x = r sin(theta), z = r cos(theta) with r = c*twtt/2.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)
    twtt = np.asarray(twtt, dtype=np.float64)
    t_half = twtt / 2.0
    dt = t_half / n_steps
    p = np.sin(theta0) / profile.at(z_start_depth)
    x = np.zeros_like(theta0 + twtt)
    z = np.full_like(x, float(z_start_depth))
    for _ in range(n_steps):
        c = profile.at(z)
        s = np.clip(p * c, -0.999999, 0.999999)
        x = x + c * s * dt
        z = z + c * np.sqrt(1.0 - s * s) * dt
    return x, z - z_start_depth, np.hypot(x, z - z_start_depth)
