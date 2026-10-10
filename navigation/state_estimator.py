"""Navigation fusion: attitude from the AHRS, horizontal position by DVL dead reckoning, depth from the pressure sensor,
with a position-uncertainty model that grows with distance travelled (and faster while bottom lock is lost)."""
import numpy as np

from preprocessing.coordinate_systems import R_enu_body
from .dvl import dead_reckon
from .pose_interpolation import PoseTrack


def estimate(nav, cfg_nav=None, sigma0_xy=0.5):
    """nav: dict with t, roll, pitch, yaw (measured), v_body, dvl_valid, depth (measured, m positive down).
    Returns dict(track=PoseTrack, positions, sigma_xy, extrapolated)."""
    cfg_nav = cfg_nav or {}
    pos, extrap = dead_reckon(nav["t"], nav["roll"], nav["pitch"], nav["yaw"], nav["v_body"], nav["dvl_valid"],
                              nav["depth"], p0_xy=tuple(nav.get("gps_start_xy", (0.0, 0.0))))
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))])
    k = float(cfg_nav.get("dvl_scale_error", 0.0015)) + 0.001                # drift fraction of distance (1-sigma-ish)
    sig = np.sqrt(sigma0_xy ** 2 + (k * dist) ** 2)
    # lost bottom lock: unobserved velocity error accumulates
    dt = np.diff(nav["t"], prepend=nav["t"][0])
    sig = np.sqrt(sig ** 2 + np.cumsum(np.where(extrap, (0.15 * dt) ** 2 * 4.0, 0.0)))
    R = np.stack([R_enu_body(r, p, y) for r, p, y in zip(nav["roll"], nav["pitch"], nav["yaw"])])
    return {"track": PoseTrack(nav["t"], pos, R), "positions": pos, "sigma_xy": sig, "extrapolated": extrap,
            "distance": dist}
