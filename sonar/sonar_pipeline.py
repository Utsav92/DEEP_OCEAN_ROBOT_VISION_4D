"""Multibeam pipeline: pings + navigation track + calibration + sound-speed profile -> filtered world soundings.

Raw pings are untouched; the result carries explicit validity masks per stage so every rejection is accountable.
"""
from dataclasses import dataclass, field

import numpy as np

from preprocessing.sonar_filter import (invalid_range_mask, quality_mask, spike_mask, incidence_mask)
from preprocessing.timestamp_sync import sync_pings
from . import multibeam as mb
from .intensity_processing import normalise


@dataclass
class SoundingSet:
    ping: np.ndarray            # (M,) int32 ping index
    beam: np.ndarray            # (M,) int16
    t: np.ndarray               # (M,) s
    xyz: np.ndarray             # (M,3) world ENU m
    xyz_sonar: np.ndarray       # (M,3) sonar frame
    rng: np.ndarray             # (M,)
    intensity: np.ndarray       # (M,) 0..1 normalised
    quality: np.ndarray         # (M,)
    sigma_h: np.ndarray         # (M,)
    sigma_v: np.ndarray         # (M,)
    T_world_vehicle: np.ndarray  # (N,4,4) per ping (navigation used)
    T_world_sonar: np.ndarray    # (N,4,4)
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)


def process(pings, calib, track, profile, cfg_sonar, sigma_xy_fn=None, clock_offset_s=0.0):
    calib.validate()
    fl = cfg_sonar["filter"]
    Tv, status = sync_pings(pings.t, track, clock_offset_s, max_gap_s=2.0)
    Ts = Tv @ calib.T_vehicle_sonar
    depth_sonar = -Ts[:, 2, 3]
    r = mb.ranges_from_pings(pings, profile, depth_sonar)
    xs = mb.sonar_frame_points(r, pings)                      # (N,B,3)
    xw = mb.world_points(xs, Ts)
    N, B = r.shape
    stats = {"pings": int(N), "beams": int(B), "soundings_total": int(N * B)}
    v0 = np.isfinite(pings.twtt) if pings.ranges_m is None else np.isfinite(pings.ranges_m)
    stats["with_detection"] = int(v0.sum())
    v1 = v0 & invalid_range_mask(r, cfg_sonar["min_range_m"], cfg_sonar["max_range_m"])
    stats["after_range_gate"] = int(v1.sum())
    v2 = v1 & quality_mask(pings.quality, fl["min_quality"])
    stats["after_quality"] = int(v2.sum())
    v2 &= incidence_mask(pings.beam_theta, fl["max_incidence_deg"])[None, :]
    stats["after_incidence"] = int(v2.sum())
    v3 = spike_mask(xw[..., 2], v2, k_sigma=fl["outlier_sigma"])
    stats["after_spike_filter"] = int(v3.sum())
    v4 = v3 & (status == 0)[:, None]
    stats["after_pose_status"] = int(v4.sum())
    sig_xy = sigma_xy_fn(pings.t) if sigma_xy_fn is not None else np.full(N, 0.5)
    sh, sv = mb.point_uncertainty(r, pings, cfg_sonar, sig_xy)
    ints = normalise(pings.intensity, v4)
    ip, ib = np.nonzero(v4)
    return SoundingSet(ping=ip.astype(np.int32), beam=ib.astype(np.int16), t=pings.t[ip], xyz=xw[ip, ib],
                       xyz_sonar=xs[ip, ib], rng=r[ip, ib], intensity=ints[ip, ib].astype(np.float32),
                       quality=pings.quality[ip, ib], sigma_h=sh[ip, ib].astype(np.float32),
                       sigma_v=sv[ip, ib].astype(np.float32), T_world_vehicle=Tv, T_world_sonar=Ts,
                       stats={**stats, "kept": int(len(ip)), "kept_fraction_of_detections": float(len(ip) / max(v0.sum(), 1)),
                              "pose_status_nonzero": int((status != 0).sum())})
