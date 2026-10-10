"""Multibeam / forward-looking sonar loaders.

* load_pings_npz: the project ping format (also written by the simulator).
* load_mblist_ascii: adapter for MB-System `mblist` ASCII exports (time, lon, lat, depth, amplitude). It is exercised
  by a unit test on a synthetic text fixture only - NO real MB-System file has been processed (MB-System is not
  installed in this environment).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pings:
    t: np.ndarray                  # (N,) s, sonar clock
    twtt: np.ndarray               # (N,B) s, NaN invalid. (Ignored when ranges_m is given.)
    beam_theta: np.ndarray         # (B,) rad across-track from nadir, + starboard
    beam_phi: np.ndarray           # (B,) rad along-track, + forward
    intensity: np.ndarray          # (N,B) normalised 0..1
    quality: np.ndarray            # (N,B) int 0..3, 0 = no detection
    c_sonar: np.ndarray            # (N,) sound speed at the transducer, m/s
    T_vehicle_sonar: np.ndarray    # (4,4)
    ranges_m: np.ndarray = None    # (N,B) calibrated slant ranges if the source provides them
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.t)

    @property
    def n_beams(self):
        return self.beam_theta.shape[0]


def load_pings_npz(path):
    d = np.load(path, allow_pickle=False)
    ranges = d["ranges_m"] if "ranges_m" in d.files else None
    return Pings(t=d["t"], twtt=d["twtt"], beam_theta=d["beam_theta"], beam_phi=d["beam_phi"],
                 intensity=d["intensity"], quality=d["quality"], c_sonar=d["c_sonar"],
                 T_vehicle_sonar=d["T_vehicle_sonar"], ranges_m=ranges,
                 meta={"source": str(path), "range_is_calibrated": bool(d["range_is_calibrated"])
                       if "range_is_calibrated" in d.files else False})


def load_mblist_ascii(text, lon0=None, lat0=None):
    """Parse whitespace separated rows `epoch_s lon lat depth_m amplitude` (mblist -OTXYzA style). Returns dict of
    arrays; geometry (beam angles) is NOT available from this export, so no calibrated sonar-frame geometry is made."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(v) for v in parts[:5]] + ([0.0] if len(parts) < 5 else []))
        except ValueError:
            continue
    a = np.array(rows, dtype=np.float64).reshape(-1, 5)
    return {"t": a[:, 0], "lon": a[:, 1], "lat": a[:, 2], "depth_m": a[:, 3], "amplitude": a[:, 4],
            "elevation_m": -a[:, 3], "geometry_available": False}
