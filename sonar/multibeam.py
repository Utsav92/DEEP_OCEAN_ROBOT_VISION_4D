"""Multibeam processing: travel time -> range -> sonar-frame XYZ -> world XYZ, with validity and uncertainty."""
import numpy as np

from preprocessing.beam_geometry import multibeam_xyz
from preprocessing.sound_speed import range_from_twtt_profile
from .uncertainty import sounding_sigma, range_sigma


def ranges_from_pings(pings, profile, depth_sonar, use_profile=True):
    """Slant ranges (N,B). Calibrated ranges supplied by the source are used as-is (no second conversion)."""
    if pings.ranges_m is not None:
        return np.array(pings.ranges_m, dtype=np.float64)
    if use_profile and profile is not None:
        return range_from_twtt_profile(pings.twtt, pings.beam_theta[None, :], depth_sonar[:, None], profile)
    return pings.c_sonar[:, None] * pings.twtt / 2.0


def sonar_frame_points(ranges, pings):
    return multibeam_xyz(ranges, pings.beam_theta, pings.beam_phi)             # (N,B,3)


def world_points(xyz_sonar, T_world_sonar):
    """P_world = T_world_sonar * P_sonar, per ping. xyz_sonar (N,B,3), T (N,4,4)."""
    return np.einsum("nij,nbj->nbi", T_world_sonar[:, :3, :3], xyz_sonar) + T_world_sonar[:, None, :3, 3]


def point_uncertainty(ranges, pings, cfg_sonar, sigma_nav_xy, sigma_nav_z=0.1, sigma_att_deg=0.1):
    sr = range_sigma(ranges, cfg_sonar["range_sigma_frac"], cfg_sonar["range_sigma_floor_m"])
    st = np.radians(cfg_sonar["angle_sigma_deg"])
    return sounding_sigma(ranges, pings.beam_theta[None, :], sr, st,
                          np.asarray(sigma_nav_xy)[:, None], sigma_nav_z, np.radians(sigma_att_deg))
