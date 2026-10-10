"""Projection helpers between world, vehicle and sonar frames and beam-fan geometry for visualisation."""
import numpy as np

from preprocessing.coordinate_systems import invert, transform_points
from preprocessing.beam_geometry import multibeam_directions


def world_to_sonar(points_world, T_world_sonar):
    return transform_points(invert(T_world_sonar), points_world)


def sonar_to_range_angle(p_sonar):
    """Across-track angle from nadir (rad, + starboard) and slant range of sonar-frame points (planar model)."""
    p = np.asarray(p_sonar)
    r = np.linalg.norm(p, axis=-1)
    theta = np.arctan2(p[..., 1], p[..., 2])
    return r, theta


def beam_fan_segments(T_world_sonar, theta, ranges):
    """Line segments (n_beams,2,3) from the sonar head to each return (world), for drawing the acoustic fan."""
    d = multibeam_directions(theta)
    R, t = T_world_sonar[:3, :3], T_world_sonar[:3, 3]
    end = t + (np.nan_to_num(ranges)[:, None] * d) @ R.T
    start = np.repeat(t[None, :], len(theta), 0)
    return np.stack([start, end], 1), np.isfinite(ranges)
