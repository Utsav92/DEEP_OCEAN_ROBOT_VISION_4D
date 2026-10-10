"""Sequence preprocessing: load pings + navigation, fuse navigation, produce filtered world soundings (dead-reckoned)."""
import json
import os

import numpy as np

from .calibration import from_pings
from .navigation_loader import load_nav_npz, describe
from .sonar_loader import load_pings_npz
from .sound_speed import SoundSpeedProfile
from .config import abspath, ensure_dir
from navigation.state_estimator import estimate
from sonar.sonar_pipeline import process


def preprocess(cfg):
    sim = abspath(cfg, cfg["paths"]["simulation"])
    proc = ensure_dir(abspath(cfg, cfg["paths"]["processed"]))
    diag = ensure_dir(os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics"))
    pings = load_pings_npz(os.path.join(sim, "pings.npz"))
    nav = load_nav_npz(os.path.join(sim, "nav_measured.npz"))
    prof = SoundSpeedProfile.from_tsd(cfg["sound_speed"]["profile"], cfg["sound_speed"]["assumed_profile_bias_mps"])
    est = estimate(nav, cfg["navigation"])
    calib = from_pings(pings)
    sig_fn = lambda t: np.interp(t, nav["t"], est["sigma_xy"])
    ss = process(pings, calib, est["track"], prof, cfg["sonar"], sigma_xy_fn=sig_fn)
    np.savez_compressed(os.path.join(proc, "soundings_dr.npz"), ping=ss.ping, beam=ss.beam, t=ss.t,
                        xyz=ss.xyz.astype(np.float32), xyz_sonar=ss.xyz_sonar.astype(np.float32),
                        rng=ss.rng.astype(np.float32), intensity=ss.intensity, quality=ss.quality,
                        sigma_h=ss.sigma_h, sigma_v=ss.sigma_v, T_world_vehicle=ss.T_world_vehicle,
                        T_world_sonar=ss.T_world_sonar, ping_t=pings.t, T_vehicle_sonar=calib.T_vehicle_sonar)
    np.savez_compressed(os.path.join(proc, "nav_dr.npz"), t=nav["t"], p=est["positions"],
                        R=est["track"].R, sigma_xy=est["sigma_xy"], extrapolated=est["extrapolated"],
                        distance=est["distance"])
    rep = {"navigation": describe(nav), "calibration_source": calib.source, "sound_speed_profile": cfg["sound_speed"]["profile"],
           "filter_stats": ss.stats, "simulated": True}
    with open(os.path.join(diag, "preprocess_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    return ss, est, rep
