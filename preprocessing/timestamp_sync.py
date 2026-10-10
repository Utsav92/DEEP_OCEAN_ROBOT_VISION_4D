"""Sensor time synchronisation: associate sonar pings with interpolated vehicle poses; report quality."""
import numpy as np


def check_monotonic(t, name="t"):
    d = np.diff(t)
    return {"name": name, "monotonic": bool(np.all(d > 0)), "n_nonpositive_steps": int(np.sum(d <= 0))}


def sort_and_dedupe(t, *arrays):
    """Sort by time and drop repeated timestamps (keeps the first). Returns (t, *arrays, kept_index)."""
    order = np.argsort(t, kind="stable")
    ts = t[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    idx = order[keep]
    return (t[idx],) + tuple(a[idx] for a in arrays) + (idx,)


def sync_pings(ping_t, track, clock_offset_s=0.0, max_gap_s=2.0):
    """Interpolate vehicle poses at ping time + offset. Returns poses (N,4,4), status (N,) 0 ok, 1 outside track, 2 gap."""
    T, status = track.at(np.asarray(ping_t) + clock_offset_s, max_gap_s=max_gap_s)
    return T, status


def estimate_clock_offset(ping_t, ping_signal, nav_t, nav_signal, search_s=2.0, step_s=0.01):
    """Cross-correlate a signal seen by both sensors (e.g. sonar-derived altitude vs vehicle depth rate) to estimate the
    clock offset. Returns the shift (s) to ADD to ping times. Assumes both signals are comparable in scale."""
    shifts = np.arange(-search_s, search_s + step_s, step_s)
    a = (ping_signal - np.mean(ping_signal)) / (np.std(ping_signal) + 1e-12)
    best, bs = -1e18, 0.0
    for s in shifts:
        b = np.interp(ping_t + s, nav_t, nav_signal)
        b = (b - b.mean()) / (b.std() + 1e-12)
        c = float(np.mean(a * b))
        if c > best:
            best, bs = c, s
    return float(bs), float(best)
