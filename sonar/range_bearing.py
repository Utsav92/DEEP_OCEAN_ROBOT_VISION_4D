"""Polar (range, bearing) <-> Cartesian conversions for imaging sonar. Bearing is positive to starboard of the boresight;
Cartesian x = forward (range * cos b), y = starboard (range * sin b)."""
import numpy as np


def polar_to_cart(rng, bearing):
    return rng * np.cos(bearing), rng * np.sin(bearing)


def cart_to_polar(x, y):
    return np.hypot(x, y), np.arctan2(y, x)


def cartesian_grid(max_range, fov_rad, size=256):
    """Pixel centres of a Cartesian image covering the sonar fan. Returns X (forward), Y (starboard) in metres."""
    half = max_range * np.sin(fov_rad / 2.0)
    ys = np.linspace(-half, half, size)
    xs = np.linspace(0.0, max_range, size)
    Y, X = np.meshgrid(ys, xs)
    return X, Y
