"""AHRS/IMU attitude model: heading bias + random walk + white noise; roll/pitch white noise (gravity-referenced)."""
import numpy as np


class ImuModel:
    def __init__(self, heading_bias_deg=0.25, heading_noise_deg=0.08, heading_walk_deg_per_sqrt_s=0.004,
                 attitude_noise_deg=0.05):
        self.hb = np.radians(heading_bias_deg)
        self.hn = np.radians(heading_noise_deg)
        self.hw = np.radians(heading_walk_deg_per_sqrt_s)
        self.an = np.radians(attitude_noise_deg)

    def measure(self, t, roll, pitch, yaw, rng):
        """Vectorised over time samples t (s). Returns measured (roll, pitch, yaw) in radians."""
        dt = np.diff(t, prepend=t[0])
        walk = np.cumsum(rng.normal(0.0, 1.0, len(t)) * self.hw * np.sqrt(np.maximum(dt, 0.0)))
        yaw_m = yaw + self.hb + walk + rng.normal(0.0, self.hn, len(t))
        return roll + rng.normal(0.0, self.an, len(t)), pitch + rng.normal(0.0, self.an, len(t)), yaw_m
