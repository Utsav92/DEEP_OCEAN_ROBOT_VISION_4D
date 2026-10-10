"""Control-panel logic (Text-DAT module in UI_CONTROLS): radio helpers, render-style parameters and the diagnostics text.
All of it except diagnostics_text() is pure python."""
import time

import numpy as np

_gpu = {"t": 0.0, "mb": 0.0}


def radio_index(values, default=1):
    for i, v in enumerate(values, start=1):
        if float(v) > 0.5:
            return i
    return default


def post_params(style, bloom_on, glow, atmosphere_on):
    """Uniforms for the post TOPs. Scientific style (0): no bloom/sparkle/scanlines, input passes through."""
    cin = float(style) > 0.5
    return {"uGlow": float(glow) if (cin and float(bloom_on) > 0.5) else (0.25 * float(glow) if float(bloom_on) > 0.5 else 0.0),
            "uRadius": 3.0 if cin else 2.0, "uThreshold": 0.55 if cin else 0.8,
            "uHaze": 1.0 if (float(atmosphere_on) > 0.5 and cin) else (0.35 if float(atmosphere_on) > 0.5 else 0.0),
            "uVignette": 0.45 if cin else 0.12, "uSparkle": 1.0 if (float(atmosphere_on) > 0.5 and cin) else 0.0,
            "uScan": 0.6 if cin else 0.0, "uChroma": 1.4 if cin else 0.0}


def coverage_at(t_first_sorted, n_cells_planned, t):
    """Fraction of the planned survey area observed by observation time t (cells with first-observation time <= t)."""
    return float(np.searchsorted(t_first_sorted, t, side="right")) / max(int(n_cells_planned), 1)


def _gpu_mb(base):
    now = time.time()
    if now - _gpu["t"] > 2.0:
        tot = 0
        stack = [base]
        while stack:
            c = stack.pop()
            for ch in c.children:
                if ch.isCOMP:
                    stack.append(ch)
                elif ch.family == "TOP":
                    try:
                        tot += int(ch.gpuMemory)
                    except Exception:
                        pass
        _gpu.update(t=now, mb=tot / 1e6)
    return _gpu["mb"]


def diagnostics_text(time_s, ping, version, mix, style, color_mode):
    """Expression entry point for the diagnostics Text TOP (arguments make TouchDesigner track the dependencies)."""
    import td
    base = td.op("/project1/DEEP_OCEAN_ROBOT_VISION_4D")
    br = base.op("DATA_INPUT/data_bridge").module
    st = br.get_store(base)
    sc = base.fetch("scene", {})
    k = int(ping)
    ends = np.asarray(st.array("beam_end_slam")[k])
    o = np.asarray(st.array("beam_origin_slam")[k], dtype=np.float64)
    valid = ends[:, 3] > 0.5
    rng = np.linalg.norm(ends[valid, :3] - o, axis=1) if valid.any() else np.zeros(1)
    T = br.pose_at(st, float(time_s), float(mix))
    depth = -(T[1, 3] + float(st.meta["z_ref_m"]))
    attr_t = base.fetch("attr_t_sorted", None)
    if attr_t is None:
        a = np.asarray(st.array("map_attr"))[..., 1].ravel()
        attr_t = np.sort(a[a < 1e8])
        base.store("attr_t_sorted", attr_t)
    n_pts = int(np.searchsorted(attr_t, float(time_s), side="right"))
    tf = base.fetch("recon_tfirst_sorted", None)
    if tf is None:
        r = np.asarray(st.array("recon_tex"))[..., 1].ravel()
        tf = np.sort(r[r < 1e8])
        base.store("recon_tfirst_sorted", tf)
    cov = coverage_at(tf, base.fetch("planned_cells", 1), float(time_s))
    loops_now = sum(1 for l in (base.fetch("loops_json") or []) if l["t_s"] <= time_s)
    sigma = float(np.interp(time_s, st.ping_t, np.asarray(st.array("ping_sigma_dr"))))
    vt = np.load(st.dir + "/versions.npz")
    nloops_v = int(vt["n_loops"][int(version)])
    meta = st.meta
    s = meta.get("slam", {})
    nav_status = "SLAM-corrected" if mix > 0.5 else "dead-reckoned (DVL+AHRS+depth)"
    cm = ("INTENSITY", "DEPTH", "AGE", "SCAN BANDS", "UNCERTAINTY")[int(np.clip(color_mode - 1, 0, 4))]
    return (
        "DEEP OCEAN ROBOT VISION - SONAR 4D\n"
        f"dataset   : {meta.get('dataset', '')[:56]}\n"
        f"data      : {meta.get('label', '')[:60]}\n"
        f"time      : t = {time_s:8.1f} s   scan {k + 1}/{st.n_pings}   map v{int(version) + 1}/{meta['n_versions']}\n"
        f"vehicle   : depth {depth:7.1f} m   nav {nav_status}\n"
        f"nav sigma : dead-reckoning 1-sigma {sigma:5.1f} m (model)\n"
        f"sonar     : valid beams {int(valid.sum()):3d}/{len(valid)}   range {rng.min():5.1f}-{rng.max():5.1f} m\n"
        f"points    : {n_pts:7d} shown of {meta['display_points']} (display subset)\n"
        f"terrain   : {sc.get('elev_min', 0):7.1f} .. {sc.get('elev_max', 0):7.1f} m (real NOAA grid)\n"
        f"coverage  : {100 * cov:5.1f} % of planned survey area\n"
        f"slam      : odometry {s.get('odometry_edges', 0)} + sonar {s.get('sonar_odometry_edges', 0)} + "
        f"loops {loops_now} (v{int(version) + 1}: {nloops_v})\n"
        f"render    : {_fps(base):4.0f} fps   GPU textures {_gpu_mb(base):6.0f} MB\n"
        f"colour    : {cm}   style {'CINEMATIC' if style > 0.5 else 'SCIENTIFIC'}")


def _fps(base):
    try:
        return float(base.op("DATA_INPUT/timeline_state")["fps"].eval())
    except Exception:
        return 0.0
