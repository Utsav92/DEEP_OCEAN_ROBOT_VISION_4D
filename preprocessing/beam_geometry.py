"""Sonar beam geometry in the SONAR frame (x along-track/forward, y across-track/starboard, z down).

Multibeam: beam i has an across-track steering angle theta_i from nadir (positive to starboard) and an along-track
angle phi_i (positive forward; zero for a flat-array sonar). The unit direction is
    d = [ sin(phi), cos(phi) sin(theta), cos(phi) cos(theta) ]
so for phi = 0 a return at range r is at (0, r sin(theta), r cos(theta)): the planar model x_across = r sin(theta),
z = r cos(theta). Depth below the head is d_z * r.

Forward-looking sonar (FLS): bearing b (positive to starboard of the boresight), elevation e below the horizontal
boresight plane, boresight tilted down by `tilt`: direction in the sonar frame
    d = [ cos(e') cos(b), cos(e') sin(b), sin(e') ] with e' = tilt + e (down positive).
"""
import numpy as np


def beam_angles(n_beams, swath_deg):
    """Equi-angular across-track beam fan, from -swath/2 (port) to +swath/2 (starboard), radians."""
    return np.radians(np.linspace(-swath_deg / 2.0, swath_deg / 2.0, n_beams))


def multibeam_directions(theta, phi=None):
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.zeros_like(theta) if phi is None else np.asarray(phi, dtype=np.float64)
    return np.stack([np.sin(phi), np.cos(phi) * np.sin(theta), np.cos(phi) * np.cos(theta)], axis=-1)


def multibeam_xyz(ranges, theta, phi=None):
    """Sonar-frame XYZ of returns. ranges (..., B), theta (B,) -> (..., B, 3)."""
    d = multibeam_directions(theta, phi)
    return np.asarray(ranges, dtype=np.float64)[..., None] * d


def fls_direction(bearing, elevation, tilt_down=0.0):
    e = np.asarray(elevation, dtype=np.float64) + tilt_down
    b = np.asarray(bearing, dtype=np.float64)
    return np.stack([np.cos(e) * np.cos(b), np.cos(e) * np.sin(b), np.sin(e)], axis=-1)


def incidence_angle(direction_sonar, normal_sonar):
    """Angle (rad) between the ray and the surface normal; 0 = looking straight onto the surface."""
    d = np.asarray(direction_sonar)
    n = np.asarray(normal_sonar)
    c = np.abs(np.sum(d * n, axis=-1))
    return np.arccos(np.clip(c, 0.0, 1.0))
