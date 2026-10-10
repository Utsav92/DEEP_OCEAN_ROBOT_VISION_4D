"""Navigation sensor loaders and validation (only available channels are reported as available)."""
import numpy as np


def load_nav_npz(path):
    d = np.load(path, allow_pickle=False)
    nav = {k: d[k] for k in d.files}
    required = ["t", "roll", "pitch", "yaw", "depth"]
    missing = [k for k in required if k not in nav]
    if missing:
        raise ValueError(f"navigation file lacks required channels: {missing}")
    if np.any(np.diff(nav["t"]) <= 0):
        raise ValueError("navigation timestamps are not strictly increasing")
    nav["channels"] = {"imu": True, "depth": True, "dvl": "v_body" in nav and bool(np.any(nav.get("dvl_valid", False))),
                       "usbl": "usbl" in nav, "reference_nav": "reference" in nav}
    return nav


def describe(nav):
    n = len(nav["t"])
    dt = np.diff(nav["t"])
    out = {"samples": int(n), "duration_s": float(nav["t"][-1] - nav["t"][0]),
           "rate_hz_median": float(1.0 / np.median(dt)), "max_gap_s": float(dt.max()),
           "channels": nav["channels"]}
    if "dvl_valid" in nav:
        out["dvl_valid_fraction"] = float(np.mean(nav["dvl_valid"]))
    return out
