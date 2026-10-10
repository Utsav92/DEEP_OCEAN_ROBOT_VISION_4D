"""Temporal-4D parameter helpers (pure python; unit-tested). The heavy lifting is on the GPU: temporal_points.vert takes
observation time, history window, decay and the artistic displacement as uniforms - nothing is a 2D feedback trail."""
import numpy as np

HISTORY_SIZES = (5, 10, 20, 50, 100, 500)
PING_PERIOD_S = 2.0

MODES = {1: "OFF", 2: "SONAR ECHO", 3: "SCAN LAYERS", 4: "TIME EXPLOSION (artistic)"}


def nearest_history(value):
    """Snap a slider value to the closest supported history size (5, 10, 20, 50, 100, 500 scans)."""
    v = float(value)
    return int(min(HISTORY_SIZES, key=lambda s: abs(np.log(s) - np.log(max(v, 1.0)))))


def history_seconds(n_scans, ping_period=PING_PERIOD_S):
    return float(n_scans) * float(ping_period)


def layer_seconds(n_scans, ping_period=PING_PERIOD_S, layers=6):
    return max(history_seconds(n_scans, ping_period) / layers, ping_period)


def uniforms(mode, n_scans, decay_s, explode_mps, ping_period=PING_PERIOD_S):
    """Uniform values for the temporal MAT given the UI state. mode: 1 off, 2 echo, 3 layers, 4 explosion."""
    m = int(round(mode))
    return {"visible": m > 1, "uHistoryS": history_seconds(n_scans, ping_period),
            "uDecay": float(decay_s) if m != 3 else history_seconds(n_scans, ping_period) * 2.0,
            "uExplodeMps": float(explode_mps) if m == 4 else 0.0,
            "uLayerSeconds": layer_seconds(n_scans, ping_period) if m == 3 else history_seconds(n_scans, ping_period),
            "n_scans": int(n_scans)}


def visible_scan_range(ping_t, t, n_scans):
    """(first, last) ping indices inside the window at observation time t - the CPU twin of the shader test."""
    k = int(np.searchsorted(ping_t, t, side="right") - 1)
    return (max(0, k - int(n_scans) + 1), k) if k >= 0 else (0, -1)
