"""SIMULATED AUV + multibeam + forward-looking sonar over REAL NOAA bathymetry (Source E).

Everything produced here is labelled simulated. The terrain under the vehicle is real gridded bathymetry; the
vehicle, sensor noise, navigation error and the few inserted seabed objects are synthetic. Ground truth is written to a
separate file that the processing pipeline never reads (only evaluation does).
"""
import json
import os

import numpy as np

from preprocessing.coordinate_systems import R_enu_body, mount_transform, transform_points
from preprocessing.beam_geometry import beam_angles, multibeam_directions, fls_direction
from preprocessing.sound_speed import SoundSpeedProfile, twtt_from_range
from navigation.imu import ImuModel
from navigation.dvl import DvlModel
from navigation.depth_sensor import DepthSensor
from navigation.pose_interpolation import PoseTrack
from .terrain import SimTerrain, choose_object_sites
from .trajectory import plan


def _texture(p, seed=7):
    """Deterministic seabed reflectivity texture in [0,1] from position (hash of a coarse lattice, bilinear)."""
    rng = np.random.default_rng(seed)
    lat = rng.random((65, 65))
    u = (p[:, 0] / 9.0) % 64.0
    v = (p[:, 1] / 9.0) % 64.0
    i0, j0 = np.clip(np.floor(u).astype(int), 0, 63), np.clip(np.floor(v).astype(int), 0, 63)
    fu, fv = u - i0, v - j0
    a = lat[j0, i0] * (1 - fu) + lat[j0, i0 + 1] * fu
    b = lat[j0 + 1, i0] * (1 - fu) + lat[j0 + 1, i0 + 1] * fu
    return a * (1 - fv) + b * fv


def simulate(cfg, grid, out_dir):
    sc, so, nc, sn = cfg["simulation"], cfg["sonar"], cfg["navigation"], cfg["sound_speed"]
    rng = np.random.default_rng(sc["seed"])
    os.makedirs(out_dir, exist_ok=True)

    # ---- truth terrain (+ flagged synthetic objects)
    half = (min(abs(grid.x[0]), grid.x[-1]) - sc["survey_inner_margin_m"],
            min(abs(grid.y[0]), grid.y[-1]) - sc["survey_inner_margin_m"])
    half = (min(half[0], sc.get("survey_half_extent_m", [1e9, 1e9])[0]), min(half[1], sc.get("survey_half_extent_m", [1e9, 1e9])[1]))
    objs = []
    oc = sc["objects"]
    for cx, cy, yaw in choose_object_sites(grid, oc["count"], sc["survey_inner_margin_m"] + 100, rng, oc["size_m"],
                                              bounds=(0.85 * half[0], 0.85 * half[1])):
        objs.append({"x": float(cx), "y": float(cy), "yaw": float(yaw),
                     "length": float(oc["size_m"][0]), "width": float(oc["size_m"][1]),
                     "height": float(rng.uniform(*oc["height_m"])), "synthetic": True})
    terrain = SimTerrain(grid, objs)
    tr = plan(terrain, sc, half)
    t = tr["t"]

    # ---- navigation sensors (measured)
    depth_true = -tr["pos"][:, 2]
    imu = ImuModel(nc["heading_bias_deg"], nc["heading_noise_deg"], nc["heading_walk_deg_per_sqrt_s"],
                   nc["attitude_noise_deg"])
    dvl = DvlModel(nc["dvl_scale_error"], nc["dvl_noise_mps"], nc["dvl_bias_mps"], sc["dvl_dropouts"])
    ds_ = DepthSensor(nc["depth_noise_m"], lat_deg=grid.frame.lat0)
    r_m, p_m, y_m = imu.measure(t, tr["roll"], tr["pitch"], tr["yaw"], rng)
    v_meas, valid = dvl.measure(t, tr["v_body"], rng)
    d_meas = ds_.measure(depth_true, rng)
    np.savez_compressed(os.path.join(out_dir, "nav_measured.npz"), t=t, roll=r_m, pitch=p_m, yaw=y_m, v_body=v_meas,
                        dvl_valid=valid, depth=d_meas,
                        gps_start_xy=tr["pos"][0, :2] + rng.normal(0.0, 3.0, 2), altitude_dvl=tr["altitude"] + rng.normal(0, 0.05, len(t)))

    # ---- multibeam pings
    prof_true = SoundSpeedProfile.from_tsd(cfg["sound_speed"]["profile"])
    truth_track = PoseTrack(t, tr["pos"], tr["R"])
    ping_t = np.arange(5.0, t[-1] - 5.0, sc["ping_period_s"]) + sc["ping_time_offset_s"]
    Tv, _ = truth_track.at(ping_t)
    T_vs = mount_transform(so["mount_xyz_m"], so["mount_rpy_deg"])
    th = beam_angles(so["n_beams"], so["swath_deg"])
    B, N = len(th), len(ping_t)
    d_nom = multibeam_directions(th)
    th_err = np.radians(so["angle_sigma_deg"]) * rng.standard_normal((N, B))
    d_act = multibeam_directions(th[None, :] + th_err)            # (N,B,3) true beam directions in the sonar frame
    Ts = Tv @ T_vs
    origins = np.repeat(Ts[:, :3, 3][:, None, :], B, 1).reshape(-1, 3)
    dirs = np.einsum("nij,nbj->nbi", Ts[:, :3, :3], d_act).reshape(-1, 3)
    r, hit, nrm = terrain.cast(origins, dirs, so["max_range_m"])
    r = r.reshape(N, B)
    hit = hit.reshape(N, B, 3)
    nrm = nrm.reshape(N, B, 3)
    inc = np.arccos(np.clip(np.abs(np.sum(nrm * dirs.reshape(N, B, 3), -1)), 0, 1))
    r = np.where(r >= so["min_range_m"], r, np.nan)
    z_s = -Ts[:, 2, 3][:, None]                                   # sonar depth
    c_mean = prof_true.harmonic_mean(z_s, z_s + np.nan_to_num(r) * np.cos(th)[None, :] * 1.0)
    sig_r = so["range_sigma_floor_m"] + so["range_sigma_frac"] * np.nan_to_num(r)
    r_noisy = r + sig_r * rng.standard_normal(r.shape)
    twtt = twtt_from_range(r_noisy, c_mean)
    # outliers: spurious early returns (water column / bubbles), delivered with an ordinary quality flag
    out_m = rng.random(r.shape) < so["outlier_prob"]
    twtt = np.where(out_m & np.isfinite(twtt), twtt * rng.uniform(0.3, 0.9, r.shape), twtt)
    drop = rng.random(r.shape) < so["dropout_prob"]
    twtt = np.where(drop, np.nan, twtt)
    # backscatter: Lambert-like incidence dependence x seabed texture, TVG-compensated, plus speckle
    tex = _texture(hit.reshape(-1, 3)).reshape(N, B)
    obj_mask = np.zeros((N, B), bool)
    for o in terrain.objects:
        dx, dy = hit[..., 0] - o["x"], hit[..., 1] - o["y"]
        c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
        u, v = dx * c + dy * s, -dx * s + dy * c
        obj_mask |= (np.abs(u) < o["length"] / 2 + 1) & (np.abs(v) < o["width"] / 2 + 1)
    base = np.cos(inc) ** 1.5 * (0.35 + 0.65 * tex) * np.where(obj_mask, 1.6, 1.0)
    speck = rng.gamma(shape=8.0, scale=1 / 8.0, size=base.shape)
    inten = np.clip(base * speck, 0, 1.5) / 1.5
    inc_deg = np.degrees(inc)
    qual = np.where(inc_deg < 55, 3, np.where(inc_deg < 72, 2, 1)).astype(np.int8)
    qual = np.where(np.isfinite(twtt), qual, 0).astype(np.int8)
    c_sonar = prof_true.at(z_s[:, 0])
    np.savez_compressed(os.path.join(out_dir, "pings.npz"), t=ping_t, twtt=twtt.astype(np.float64),
                        intensity=inten.astype(np.float32), quality=qual, beam_theta=th, beam_phi=np.zeros(B),
                        c_sonar=c_sonar, T_vehicle_sonar=T_vs, range_is_calibrated=False)
    # ---- forward-looking sonar images (polar range x bearing), SIMULATED
    fl = so["fls"]
    fls_t, fls_img = [], []
    if fl.get("enabled", True):
        nb, nr = fl["n_bearings"], fl["n_range_bins"]
        bear = np.radians(np.linspace(-fl["fov_deg"] / 2, fl["fov_deg"] / 2, nb))
        elev = np.radians(np.linspace(-fl["vertical_aperture_deg"] / 2, fl["vertical_aperture_deg"] / 2, 5))
        tilt = np.radians(fl["tilt_down_deg"])
        edges = np.linspace(0, fl["max_range_m"], nr + 1)
        for k in range(0, N, fl["every_n_pings"]):
            Bm, Em = np.meshgrid(bear, elev)
            ds = fls_direction(Bm.ravel(), Em.ravel(), tilt)
            dw = ds @ Ts[k, :3, :3].T
            o = np.repeat(Ts[k:k + 1, :3, 3], len(dw), 0)
            rr, hp, nn = terrain.cast(o, dw, fl["max_range_m"], step=2.0, bisect=8)
            img = np.zeros((nr, nb), np.float32)
            ok = np.isfinite(rr)
            if ok.any():
                cosi = np.abs(np.sum(nn * dw, -1))
                val = (cosi ** 1.2 * (0.3 + 0.7 * _texture(hp)))[ok]
                ib = np.repeat(np.arange(nb)[None, :], len(elev), 0).ravel()[ok]
                ir = np.clip(np.searchsorted(edges, rr[ok]) - 1, 0, nr - 1)
                np.add.at(img, (ir, ib), val)
            img *= rng.gamma(6.0, 1 / 6.0, img.shape)
            fls_img.append(np.clip(img / max(img.max(), 1e-6) * 255, 0, 255).astype(np.uint8))
            fls_t.append(ping_t[k])
        np.savez_compressed(os.path.join(out_dir, "fls.npz"), t=np.array(fls_t), images=np.array(fls_img),
                            bearings=bear, max_range_m=fl["max_range_m"], tilt_down_rad=tilt, simulated=True)

    # ---- ground truth (evaluation only)
    np.savez_compressed(os.path.join(out_dir, "truth.npz"), t=t, p=tr["pos"], R=tr["R"], ping_t=ping_t,
                        ping_T=Tv, hit_enu=hit.astype(np.float32), altitude=tr["altitude"])
    np.savez_compressed(os.path.join(out_dir, "truth_terrain.npz"), x=grid.x, y=grid.y, z=terrain.z.astype(np.float32),
                        valid=terrain.valid)
    meta = {"label": cfg["simulation"]["label"], "simulated": True, "seed": sc["seed"],
            "n_pings": int(N), "n_beams": int(B), "duration_s": float(t[-1]), "track_length_m": float(
                np.sum(np.linalg.norm(np.diff(tr["pos"], axis=0), axis=1))),
            "half_extent_m": [float(half[0]), float(half[1])], "objects": objs,
            "dvl_dropouts": sc["dvl_dropouts"], "valid_beam_fraction": float(np.isfinite(twtt).mean()),
            "min_altitude_m": float(tr["altitude"].min()), "max_pitch_deg": float(np.degrees(np.abs(tr["pitch"]).max())),
            "note": "Real NOAA NCEI terrain; vehicle, sensors, noise, navigation error and objects are synthetic."}
    with open(os.path.join(out_dir, "sim_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta
