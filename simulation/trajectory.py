"""Survey trajectory planning for the simulated AUV: lawn-mower lines + tie lines, rounded turns, terrain-following depth."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from preprocessing.coordinate_systems import R_enu_body


def lawnmower_waypoints(half_x, half_y, spacing):
    xs = np.arange(-half_x, half_x + 1e-6, spacing)
    wp = []
    for i, x in enumerate(xs):
        ys = (-half_y, half_y) if i % 2 == 0 else (half_y, -half_y)
        wp += [(x, ys[0]), (x, ys[1])]
    # two tie lines across the survey (these create the loop closures a lawn-mower alone gives only between neighbours)
    last_y = wp[-1][1]
    ty = [0.55 * half_y, -0.55 * half_y] if last_y > 0 else [-0.55 * half_y, 0.55 * half_y]
    wp += [(half_x, ty[0]), (-half_x, ty[0]), (-half_x, ty[1]), (half_x, ty[1])]
    return np.array(wp, dtype=np.float64)


def densify(wp, ds):
    seg = np.diff(wp, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(L)])
    sq = np.arange(0.0, s[-1], ds)
    x = np.interp(sq, s, wp[:, 0])
    y = np.interp(sq, s, wp[:, 1])
    return np.stack([x, y], 1)


def plan(terrain, cfg_sim, half_extent):
    """Returns dict with t, pos (N,3 ENU), roll, pitch, yaw, v_enu, R (N,3,3) at nav_rate_hz (ground truth)."""
    speed = cfg_sim["speed_mps"]
    dt = 1.0 / cfg_sim["nav_rate_hz"]
    ds = speed * dt
    wp = lawnmower_waypoints(half_extent[0], half_extent[1], cfg_sim["line_spacing_m"])
    xy = densify(wp, ds)
    sig = max(cfg_sim["turn_smoothing_m"] / ds, 1.0)
    # reflect-pad so the end points are not pulled inward
    xy = np.stack([gaussian_filter1d(xy[:, k], sig, mode="nearest") for k in range(2)], 1)
    cut = int(2 * sig)
    xy = xy[cut:-cut] if cut > 0 else xy      # smoothing pins the end points: drop the bunched-up samples
    n = len(xy)
    seabed = terrain.height(xy[:, 0], xy[:, 1])
    alt, min_alt = cfg_sim["altitude_m"], cfg_sim["min_altitude_m"]
    win = int(round(130.0 / ds)) | 1
    z = maximum_filter1d(seabed, size=win, mode="nearest") + alt
    maxdz = np.tan(np.radians(cfg_sim["max_pitch_deg"])) * 0.85 * ds
    for _ in range(3):
        for i in range(1, n):
            z[i] = max(z[i], z[i - 1] - maxdz)
        for i in range(n - 2, -1, -1):
            z[i] = max(z[i], z[i + 1] - maxdz)
        z = gaussian_filter1d(z, 3.0, mode="nearest")
        z = np.maximum(z, seabed + min_alt)
    # the vehicle must stay below the sea surface
    z = np.minimum(z, -5.0)
    pos = np.column_stack([xy, z])
    t = np.arange(n) * dt
    v = np.gradient(pos, dt, axis=0)
    yaw = np.arctan2(v[:, 0], v[:, 1])                     # clockwise from north
    yaw = np.unwrap(yaw)
    pitch = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))
    yawrate = np.gradient(yaw, dt)
    roll = np.clip(0.5 * np.arctan(speed * yawrate / 9.81), -0.15, 0.15)
    R = np.stack([R_enu_body(r, p, y) for r, p, y in zip(roll, pitch, yaw)])
    v_body = np.einsum("nji,nj->ni", R, v)                  # R^T v
    return {"t": t, "pos": pos, "roll": roll, "pitch": pitch, "yaw": yaw, "v_enu": v, "v_body": v_body, "R": R,
            "altitude": z - seabed, "waypoints": wp}
