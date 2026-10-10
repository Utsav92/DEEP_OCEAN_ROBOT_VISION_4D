"""Scan echoes (CINEMATIC, not measured): an expanding luminous ring around the sonar position of each recent ping.
The ring radius is a stylised function of age and does NOT represent a recorded wavefront."""
import numpy as np


def echo_radius(age_s, speed_mps=60.0, max_radius=260.0):
    return np.clip(np.asarray(age_s) * speed_mps, 0.0, max_radius)


def echo_alpha(age_s, life_s=4.0):
    a = np.asarray(age_s, dtype=np.float64)
    return np.where((a >= 0) & (a <= life_s), (1.0 - a / life_s) ** 2, 0.0)
