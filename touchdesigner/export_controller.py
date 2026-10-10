"""Cinematic export (Text-DAT module in OUTPUT): deterministic frame-by-frame renderer for the 30 s film.

`storyboard(t)` is a pure function of video time (unit-tested): data time, control values, camera mode/path parameter and
the on-screen titles. render_frame() applies it through the same panel widgets a user would click, cooks the network and
saves a PNG; scripts/export_video.py encodes the PNG sequence to H.264 with ffmpeg. Cinematic wavefront / echo / explosion
effects are stylised and are labelled as such by the SCENE TITLES; recorded measurements are never altered.
"""
import os

import numpy as np

BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
FPS = 30
DURATION = 30.0

SCENES = [  # start, end, title, subtitle
    (0.0, 4.0, "DEEP OCEAN ROBOT VISION 4D", "descent into the dark"),
    (4.0, 8.0, "ACOUSTIC SPATIAL PERCEPTION", "calibrated multibeam returns around the vehicle"),
    (8.0, 13.0, "BATHYMETRIC RECONSTRUCTION", "point clouds accumulate into a metric seabed"),
    (13.0, 18.0, "UNDERWATER NAVIGATION AND SLAM", "dead reckoning vs SLAM, verified loop closures"),
    (18.0, 24.0, "X / Y / Z / TIME", "historical scans separate into spatial layers (artistic time axis)"),
    (24.0, 27.0, "PERSISTENT OCEAN MEMORY", "fly-through of the reconstructed seabed"),
    (27.0, 30.0, "DEEP OCEAN ROBOT VISION - SONAR 4D", "EXPLORING THE INVISIBLE OCEAN"),
]


def _ease(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def scene_index(t):
    for i, (a, b, *_r) in enumerate(SCENES):
        if a <= t < b:
            return i
    return len(SCENES) - 1


def storyboard(t, duration_data):
    """Everything that changes over the film as a function of video time t (s). `duration_data` = survey duration (s)."""
    i = scene_index(t)
    a, b, title, sub = SCENES[i]
    u = (t - a) / (b - a)
    D = duration_data
    s = dict(style_mode=1, color_mode=2, temporal_mode=1, camera_mode=2, scan_history=50, slam_mix=1.0, fade=1.0,
             show_sonar=1, show_beams=1, show_terrain=1, show_recon=1, show_traj=1, show_dr=0, show_loops=0,
             show_vehicle=1, show_versions=0, show_objects=0, show_nav_unc=0, exaggeration=2.5, glow=1.2, gain=1.2)
    cam_u = 0.0
    if i == 0:                                   # descent: the vehicle appears out of the dark, lit by its own light
        data_t = 0.0 + 90.0 * u
        s.update(camera_mode=2, show_terrain=0, show_recon=0, show_sonar=1, show_beams=0, show_nav_unc=1)
        s["fade"] = _ease(u * 2.0)
    elif i == 1:                                 # sonar activation
        data_t = 120.0 + 480.0 * u
        s.update(camera_mode=2, show_terrain=1, show_recon=0, color_mode=1)
    elif i == 2:                                 # reconstruction grows
        data_t = 600.0 + 0.40 * D * _ease(u)
        s.update(camera_mode=3, color_mode=2, show_objects=1)
    elif i == 3:                                 # SLAM
        data_t = 0.55 * D + 0.40 * D * _ease(u)
        s.update(camera_mode=5, show_dr=1, show_loops=1, show_nav_unc=1, slam_mix=_ease(u * 1.6), show_versions=0)
    elif i == 4:                                 # time explosion
        data_t = D * 0.995
        s.update(camera_mode=3, temporal_mode=4, scan_history=500, show_dr=0, show_loops=1, color_mode=4, explode_mps=0.45)
    elif i == 5:                                 # fly-through
        data_t = D
        s.update(camera_mode=6, temporal_mode=1, show_loops=0)
        cam_u = 0.15 + 0.6 * u
    else:                                        # pull back
        data_t = D
        s.update(camera_mode=6, temporal_mode=1, show_loops=1, show_dr=1)
        cam_u = 0.75 + 0.25 * u
        s["fade"] = 1.0 - _ease((t - 29.0) / 1.0) if t > 29.0 else 1.0
    alpha = min(1.0, (t - a) / 0.5) * min(1.0, (b - t) / 0.5) if i not in (6,) else min(1.0, (t - a) / 0.5)
    s.update(data_time=float(min(max(data_t, 0.0), D)), cam_u=float(cam_u), scene=i, title=title, subtitle=sub,
             title_alpha=float(max(alpha, 0.0)))
    return s


# ------------------------------------------------------------------------------------------ TouchDesigner side
def _env():
    import td
    base = td.op(BASE)
    return dict(base=base, panel=base.op("UI_CONTROLS/panel"), mc=base.op("UI_CONTROLS/main_controls"),
                tl=base.op("DATA_INPUT/timeline_controller").module, cc=base.op("CAMERA_SYSTEM/camera_controller").module,
                out=base.op("OUTPUT/OUT"), render=base.op("OUTPUT/render3d"), title=base.op("OUTPUT/title_overlay"))


def _set_const(env, name, value):
    mc = env["mc"]
    mc.seq.const[[c.name for c in mc.chans()].index(name)].par.value = value


TOGGLES = ("show_sonar", "show_beams", "show_terrain", "show_recon", "show_traj", "show_dr", "show_loops", "show_vehicle",
           "show_versions", "show_objects", "show_nav_unc")
RADIOS = {"style_mode": ("r_style", 1), "color_mode": ("r_color", 0), "temporal_mode": ("r_temporal", 0),
          "camera_mode": ("r_camera", 0)}


def apply(env, s):
    panel = env["panel"]
    for k in TOGGLES:
        w = panel.op("t_" + k)
        if w is not None:
            w.par.value0 = int(s[k])
    for k in ("scan_history", "slam_mix", "exaggeration", "glow", "gain"):
        w = panel.op("s_" + k)
        if w is not None:
            w.par.value0 = float(s[k])
    for k, (prefix, offset) in RADIOS.items():
        idx = int(s[k]) + offset if k == "style_mode" else int(s[k])
        want = idx + (1 if k == "style_mode" else 0)
        for i in range(1, 9):                            # set every radio value directly (click() is unreliable in one call)
            w = panel.op("%s%d" % (prefix, i))
            if w is not None:
                w.par.value0 = 1 if i == want else 0
    for k in ("fade", "title_alpha"):
        if k in s:
            _set_const(env, k, float(s[k]))
    if "explode_mps" in s:
        w = panel.op("s_explode_mps")
        if w is not None:
            w.par.value0 = float(s["explode_mps"])
        else:
            _set_const(env, "explode_mps", float(s["explode_mps"]))


DIAG_EXPR = "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/UI_CONTROLS/ui_controller').module.diagnostics_text("             "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/DATA_INPUT/timeline_state')['time'].eval(), "             "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/DATA_INPUT/timeline_state')['ping'].eval(), "             "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/DATA_INPUT/timeline_state')['version'].eval(), "             "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/UI_CONTROLS/main_controls')['slam_mix'].eval(), "             "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/UI_CONTROLS/main_controls')['style_mode'].eval(), "             "op('/project1/DEEP_OCEAN_ROBOT_VISION_4D/UI_CONTROLS/main_controls')['color_mode'].eval())"


def begin(hide_diagnostics=True):
    env = _env()
    env["tl"].pause()
    if hide_diagnostics:                           # the film carries titles only; the dashboard text stays in the app
        env["base"].op("OUTPUT/diagnostics_text").par.text = ""
    return {"duration_data_s": env["tl"].state()["duration"], "frames": int(FPS * DURATION)}


def end():
    env = _env()
    env["cc"].clear_override()
    env["base"].op("OUTPUT/diagnostics_text").par.text.expr = DIAG_EXPR
    _set_const(env, "fade", 1.0)
    _set_const(env, "title_alpha", 0.0)
    return "restored"


def render_frame(n, out_dir):
    env = _env()
    t = n / FPS
    D = env["tl"].state()["duration"]
    s = storyboard(t, D)
    env["tl"].set_time(s["data_time"])
    apply(env, s)
    env["title"].par.text = s["title"] + "\n" + s["subtitle"]
    if int(s["camera_mode"]) == 6:                    # scripted path parameter from the storyboard (not the data clock)
        W, fov = env["cc"].target_pose(env["base"], 6, s["data_time"], s["cam_u"])
        env["cc"].set_override(W, fov)
    else:
        env["cc"].clear_override()
    env["cc"].update()
    env["render"].cook(force=True)
    env["out"].cook(force=True)
    os.makedirs(out_dir, exist_ok=True)
    env["out"].save(os.path.join(out_dir, "frame_%04d.png" % n))
    return s["scene"]


def render_range(a, b, out_dir):
    import time
    t0 = time.perf_counter()
    scenes = [render_frame(n, out_dir) for n in range(a, b)]
    dt = time.perf_counter() - t0
    return "rendered %d..%d in %.1f s (%.0f ms/frame) scenes %s" % (a, b - 1, dt, dt / max(b - a, 1) * 1000,
                                                                   sorted(set(scenes)))
