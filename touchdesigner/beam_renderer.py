"""Acoustic beam fan of the CURRENT ping (Script SOP helper): triangles from the sonar head to consecutive valid returns.
Geometry is measured (SLAM-corrected sonar origin and returns). Point colour: r = range fraction, g = beam index."""
import numpy as np

BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
_cache = {"key": None}


def fan_triangles(origin, ends, theta, swath_deg, max_gap=2):
    """Pure geometry (tested): list of (a, b) index pairs of consecutive valid beams inside the shown swath, and the
    boolean beam mask. A missing beam wider than `max_gap` breaks the fan instead of bridging a hole."""
    ok = (ends[:, 3] > 0.5) & (np.abs(theta) <= np.radians(swath_deg) / 2.0)
    idx = np.flatnonzero(ok)
    pairs = [(int(a), int(b)) for a, b in zip(idx[:-1], idx[1:]) if b - a <= max_gap]
    return pairs, ok


def build(scriptOp, base, k, swath_deg):
    key = (int(k), round(float(swath_deg), 1))
    if _cache["key"] == key:
        return 0
    _cache["key"] = key
    scriptOp.clear()
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    ends = np.asarray(st.array("beam_end_slam")[k])
    o = np.asarray(st.array("beam_origin_slam")[k], dtype=np.float64)
    th = np.asarray(st.array("beam_theta"))
    pairs, ok = fan_triangles(o, ends, th, swath_deg)
    B = len(th)
    for a, b in pairs:
        poly = scriptOp.appendPoly(3, closed=True, addPoints=True)
        for j, (p, rf, bi) in enumerate(((o, 0.0, a / B), (ends[a, :3], 1.0, a / B), (ends[b, :3], 1.0, b / B))):
            pt = poly[j].point
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
            try:
                pt.color = (rf, bi, 1.0, 1.0)
            except Exception:
                pass
    return len(pairs)
