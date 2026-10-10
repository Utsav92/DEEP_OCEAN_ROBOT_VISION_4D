"""Observation-time controller (Text-DAT module in DATA_INPUT).

Wraps temporal_4d.time_scrubber.TimeScrubber (pure python, unit-tested) and publishes its state to the
`timeline_state` Constant CHOP every frame. Time is DATA time (seconds since the first ping); the camera is independent
of it, so freezing time never freezes camera navigation.
"""
import sys
import time

import numpy as np

BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
CHANNELS = ("time", "ping", "playing", "frozen", "speed", "direction", "version", "duration", "fps")
_g = {"scrub": None, "last": None, "fps": 30.0, "ticks": 0}


def _base():
    import td
    return td.op(BASE)


def _store():
    base = _base()
    return base.op("DATA_INPUT/data_bridge").module.get_store(base)


def _scrub():
    if _g["scrub"] is None:
        base = _base()
        proj = base.fetch("project_dir")
        if proj not in sys.path:
            sys.path.insert(0, proj)
        from temporal_4d.time_scrubber import TimeScrubber
        st = _store()
        vt = np.asarray(np.load(st.dir + "/versions.npz")["t_end"])
        _g["scrub"] = TimeScrubber(st.ping_t, vt, speed=float(base.fetch("default_speed", 60.0)))
    return _g["scrub"]


def _publish():
    import td
    s = _scrub()
    ctl = _base().op("DATA_INPUT/timeline_state")
    ver = int(np.clip(np.searchsorted(s.version_t, s.t, side="right") - 1, 0, len(s.version_t) - 1))
    vals = (s.t - s.t0, s.scan_index(), 1.0 if s.playing else 0.0, 1.0 if s.frozen else 0.0, s.speed, s.direction,
            ver, s.t1 - s.t0, _g["fps"])
    for i, v in enumerate(vals):
        setattr(ctl.par, "value%d" % i, float(v))


def tick():
    """Once per TouchDesigner frame (Execute DAT onFrameStart)."""
    now = time.perf_counter()
    dt = 0.0 if _g["last"] is None else min(now - _g["last"], 0.25)
    _g["last"] = now
    if dt > 0:
        _g["fps"] = 0.9 * _g["fps"] + 0.1 * (1.0 / dt)
    _g["ticks"] += 1
    s = _scrub()
    try:
        s.speed = max(0.0, float(_base().op("UI_CONTROLS/main_controls")["playback_speed"].eval()))
    except Exception:
        pass
    s.tick(dt)
    _publish()


def _abs_t(t_rel):                      # scrubber works in absolute ping_t (here ping_t is already relative to ping 0)
    return float(t_rel)


def play():       _scrub().play(); _publish()
def reverse():    _scrub().reverse(); _publish()
def pause():      _scrub().pause(); _publish()
def freeze(on=True): _scrub().freeze(on); _publish()
def reset():      _scrub().reset(); _publish()
def step(n=1):    _scrub().step(n); _publish()
def jump_to_scan(k):  _scrub().jump_to_scan(int(k)); _publish()
def jump_to_version(v): _scrub().jump_to_version(int(v)); _publish()
def set_time(t):  _scrub().set_time(float(t)); _publish()
def state():
    s = _scrub()
    return {"t": s.t - s.t0, "scan": s.scan_index(), "playing": s.playing, "frozen": s.frozen, "direction": s.direction,
            "speed": s.speed, "duration": s.t1 - s.t0}
