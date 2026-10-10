"""DEEP OCEAN ROBOT VISION 4D - TouchDesigner project builder (idempotent).

Run INSIDE TouchDesigner (Textport, or through the HTTP exec bridge):

    DORV_DIR = r'C:\\Users\\Utsav\\DEEP_OCEAN_ROBOT_VISION_4D'
    exec(open(DORV_DIR + r'\\touchdesigner\\build_project.py', encoding='utf-8').read())

Rebuilds /project1/DEEP_OCEAN_ROBOT_VISION_4D from scratch (the old network is destroyed first, so re-running never
duplicates operators) and, when DORV_SAVE is not False, saves Deep_Ocean_Robot_Vision_4D.toe next to this file.
Prints a BUILD REPORT; anything that did not apply is a WARNING, never silently skipped.
"""
import json
import os

import numpy as np

PROJECT_DIR = globals().get("DORV_DIR", r"C:\Users\Utsav\DEEP_OCEAN_ROBOT_VISION_4D")
SAVE = globals().get("DORV_SAVE", True)
TD_DIR = os.path.join(PROJECT_DIR, "touchdesigner")
SHADER_DIR = os.path.join(PROJECT_DIR, "shaders")
DATA_DIR = os.path.join(PROJECT_DIR, "output", "pointclouds")
ROOT_PATH = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
REPORT, WARN = [], []


def say(m):
    REPORT.append(m)
    print(m)


def warn(m):
    WARN.append(m)
    print("WARNING:", m)


# ------------------------------------------------------------------------------------------------- helpers
def fresh(parent, optype, name):
    old = parent.op(name)
    if old is not None:
        old.destroy()
    return parent.create(optype, name)


def text_dat(parent, name, path=None, text=None):
    d = fresh(parent, textDAT, name)
    if path:
        d.par.file = path
        d.par.loadonstart = True
        d.par.syncfile = True
        d.par.loadonstartpulse.pulse()
    if text is not None:
        d.text = text
    return d


def setp(o, **kw):
    for k, v in kw.items():
        try:
            setattr(o.par, k, v)
        except Exception as e:
            warn(f"{o.path}.{k}={v!r}: {e}")


def set_vec(mat, i, name, *vals):
    try:
        if mat.seq.vec.numBlocks <= i:
            mat.seq.vec.numBlocks = i + 1
        blk = mat.seq.vec[i]
        blk.par.name = name
        for k, v in zip("xyzw", vals):
            getattr(blk.par, "value" + k).val = v
    except Exception as e:
        warn(f"{mat.name}: uniform {name}: {e}")


def vec_expr(mat, i, comps):
    for k, expr in zip("xyzw", comps):
        try:
            getattr(mat.seq.vec[i].par, "value" + k).expr = expr
        except Exception as e:
            warn(f"{mat.name}: expr {expr}: {e}")


def set_sampler(mat, i, name, top):
    try:
        if mat.seq.sampler.numBlocks <= i:
            mat.seq.sampler.numBlocks = i + 1
        blk = mat.seq.sampler[i]
        blk.par.name = name
        blk.par.top = top
    except Exception as e:
        warn(f"{mat.name}: sampler {name}: {e}")


class Uniforms:
    """Sequential uniform allocator for a GLSL MAT/TOP: u.f('uGain', "expr or number"), u.v3(...)."""

    def __init__(self, op_):
        self.op, self.i = op_, 0

    def f(self, name, value):
        set_vec(self.op, self.i, name, 0.0)
        if isinstance(value, str):
            vec_expr(self.op, self.i, [value])
        else:
            set_vec(self.op, self.i, name, float(value))
        self.i += 1

    def v3(self, name, exprs):
        set_vec(self.op, self.i, name, 0.0, 0.0, 0.0)
        if all(isinstance(e, str) for e in exprs):
            vec_expr(self.op, self.i, list(exprs))
        else:
            set_vec(self.op, self.i, name, *[float(e) for e in exprs])
        self.i += 1

    def v4(self, name, vals):
        set_vec(self.op, self.i, name, *[float(e) for e in vals])
        self.i += 1


def make_mat(parent, name, vert, frag):
    m = fresh(parent, glslMAT, name)
    vd = text_dat(parent, name + "_vert", os.path.join(SHADER_DIR, vert))
    pd = text_dat(parent, name + "_frag", os.path.join(SHADER_DIR, frag))
    m.par.vdat = vd
    m.par.pdat = pd
    return m


def script_top(parent, name, body):
    cb = text_dat(parent, name + "_callbacks", text=(
        "def onSetupParameters(scriptOp):\n    return\n\ndef onPulse(par):\n    return\n\n"
        "def onCook(scriptOp):\n" + body + "\n    return\n"))
    t = fresh(parent, scriptTOP, name)
    t.par.callbacks = cb
    try:
        t.par.format = "rgba32float"
    except Exception as e:
        warn(f"{name}: format: {e}")
    return t


def grid_geo(parent, name, cols, rows, sx, sy, cx=0.0, cz=0.0, points=False):
    """Flat Grid SOP (uv 0..1) whose bounds cover the displayed region, as a Geometry COMP. points=True converts the
    vertices to point sprites (particle primitives)."""
    geo = fresh(parent, geometryCOMP, name)
    for c in list(geo.children):
        c.destroy()
    g = geo.create(gridSOP, "grid1")
    setp(g, rows=rows, cols=cols, sizex=sx, sizey=sy, orient="zx" if "zx" in [m for m in g.par.orient.menuNames] else g.par.orient.eval(),
         texture="rowcol", tx=cx, tz=cz)
    try:
        g.par.surftype = "points" if False else "triangles"
    except Exception as e:
        warn(f"{name}: surftype: {e}")
    if points:
        conv = geo.create(convertSOP, "to_points")
        conv.inputConnectors[0].connect(g)
        setp(conv, totype="part", prtype="pointsprites")
        conv.display = conv.render = True
        g.display = g.render = False
    else:
        g.display = g.render = True
    return geo


def line_mat(parent, name, rgb, width=2.0, alpha=1.0):
    m = fresh(parent, lineMAT, name)
    for nm, v in (("widthnear", width), ("widthfar", width), ("linenearcolorr", rgb[0]), ("linenearcolorg", rgb[1]),
                  ("linenearcolorb", rgb[2]), ("linenearalpha", alpha), ("linefarcolorr", rgb[0]),
                  ("linefarcolorg", rgb[1]), ("linefarcolorb", rgb[2]), ("linefaralpha", alpha)):
        try:
            setattr(m.par, nm, v)
        except Exception as e:
            warn(f"{name}.{nm}: {e}")
    return m


def script_sop_geo(parent, name, body, mat=None):
    geo = fresh(parent, geometryCOMP, name)
    for c in list(geo.children):
        c.destroy()
    cb = text_dat(geo, "callbacks", text=(
        "def onSetupParameters(scriptOp):\n    return\n\ndef onCook(scriptOp):\n" + body + "\n    return\n"))
    sop = geo.create(scriptSOP, "sop")
    sop.par.callbacks = cb
    sop.display = sop.render = True
    if mat is not None:
        geo.par.material = mat
    return geo, sop


# ================================================================================================ build
def build():
    for need in ("td_meta.json", "map_points_slam.npy", "recon_tex.npy", "loops.json", "objects.json", "versions.npz"):
        if not os.path.exists(os.path.join(DATA_DIR, need)):
            raise FileNotFoundError(f"{need} missing in {DATA_DIR}: run scripts/build_map.py first")
    meta = json.load(open(os.path.join(DATA_DIR, "td_meta.json")))
    cfg_cam_far = 80000.0

    proj = op("/project1") or root.create(baseCOMP, "project1")
    rv = fresh(proj, baseCOMP, "DEEP_OCEAN_ROBOT_VISION_4D")
    rv.store("project_dir", PROJECT_DIR)
    rv.store("data_dir", DATA_DIR)
    rv.store("default_speed", 60.0)
    P = lambda rel: f"{ROOT_PATH}/{rel}"
    MC = lambda n: f"op('{P('UI_CONTROLS/main_controls')}')['{n}'].eval()"
    TLS = lambda n: f"op('{P('DATA_INPUT/timeline_state')}')['{n}'].eval()"

    # ---------------------------------------------------------------- DATA_INPUT
    di = fresh(rv, baseCOMP, "DATA_INPUT")
    br = text_dat(di, "data_bridge", os.path.join(TD_DIR, "data_bridge.py"))
    tlm = text_dat(di, "timeline_controller", os.path.join(TD_DIR, "timeline_controller.py"))
    tls = fresh(di, constantCHOP, "timeline_state")
    names = ("time", "ping", "playing", "frozen", "speed", "direction", "version", "duration", "fps")
    try:
        tls.seq.const.numBlocks = len(names)
        for i, nm in enumerate(names):
            tls.seq.const[i].par.name = nm
            tls.seq.const[i].par.value = 0.0
    except Exception:
        for i, nm in enumerate(names):
            setp(tls, **{f"name{i}": nm, f"value{i}": 0.0})
    # scene numbers used by cameras/diagnostics
    bm = text_dat(rv, "bathymetry_renderer", os.path.join(TD_DIR, "bathymetry_renderer.py"))
    scene = bm.module.scene_info(rv)
    rv.store("scene", scene)
    rv.store("loops_json", json.load(open(os.path.join(DATA_DIR, "loops.json"))))
    rv.store("planned_cells", int(meta.get("planned_cells", 1)))
    say(f"scene: floor y {scene['floor_y']:.1f}, top y {scene['top_y']:.1f}, z_ref {scene['z_ref']:.0f} m, "
        f"{meta['n_pings']} pings, {meta['display_points']} display points, {meta['n_loops']} loop closures")

    # inventories (Table DATs) - the loader operators the spec asks for
    def table(parent, name, rows):
        t = fresh(parent, tableDAT, name)
        t.clear()
        for r in rows:
            t.appendRow([str(c) for c in r])
        return t
    table(di, "sonar_loader", [["file", "shape", "role"],
                               ["map_points_dr.npy", "512x512x4", "dead-reckoned sounding positions (a = slant range)"],
                               ["map_points_slam.npy", "512x512x4", "SLAM-corrected sounding positions"],
                               ["map_attr.npy", "512x512x4", "intensity, time, sigma_h, ping index"],
                               ["beam_end_slam.npy", f"{meta['n_pings']}x96x4", "per-ping returns (measured geometry)"]])
    table(di, "navigation_loader", [["file", "role"], ["traj_dr.npy", "dead-reckoned path"], ["traj_slam.npy", "SLAM path"],
                                    ["traj_truth.npy", "SIMULATED ground truth (evaluation overlay only)"],
                                    ["vehicle_T_slam.npy", "per-ping vehicle pose"], ["ping_sigma_dr.npy", "DR 1-sigma"]])
    table(di, "calibration_loader", [["item", "value"], ["T_vehicle_sonar", "from pings.npz (config mount 0.6,0,0.35 m)"],
                                     ["beams", "96, +-65 deg across-track"], ["sound speed", "Mackenzie profile (config.yaml)"]])
    sc_top = script_top(di, "bathymetry_loader",
                        f"    base = op('{ROOT_PATH}')\n    op('{ROOT_PATH}/bathymetry_renderer').module.cook(scriptOp, 'survey')")
    say("DATA_INPUT built")

    # ---------------------------------------------------------------- UI_CONTROLS (values first; widgets later)
    ui = fresh(rv, baseCOMP, "UI_CONTROLS")
    text_dat(ui, "ui_controller", os.path.join(TD_DIR, "ui_controller.py"))
    mc = fresh(ui, constantCHOP, "main_controls")
    controls = [
        ("show_sonar", 1), ("show_beams", 1), ("show_terrain", 1), ("show_recon", 1), ("show_wire", 0),
        ("show_vehicle", 1), ("show_traj", 1), ("show_dr", 1), ("show_truth_traj", 0), ("show_nav_unc", 1),
        ("show_loops", 1), ("show_objects", 1), ("show_truth_objects", 0), ("show_versions", 0), ("show_contours", 1),
        ("show_grid", 1), ("show_atmosphere", 1), ("show_bloom", 1),
        ("point_size", 3.0), ("opacity", 1.0), ("gain", 1.0), ("max_range", 260.0), ("swath_deg", 130.0),
        ("scan_history", 50.0), ("temporal_decay", 40.0), ("exaggeration", 2.0), ("depth_scale", 1.0),
        ("camera_speed", 260.0), ("playback_speed", 60.0), ("glow", 0.9), ("density", 1.0), ("slam_mix", 1.0),
        ("jump_scan", 0.0), ("jump_version", 0.0), ("explode_mps", 0.5),
        ("style_mode", 1), ("color_mode", 2), ("temporal_mode", 1), ("camera_mode", 1), ("terrain_view", 1),
        ("cam_fov", 60.0), ("cam_blend_s", 1.2), ("chase_back", 160.0), ("chase_up", 70.0), ("orbit_radius", 1900.0),
        ("orbit_height", 800.0), ("orbit_period", 120.0), ("overhead_height", 3200.0),
        ("clock", 0.0), ("fade", 1.0), ("title_alpha", 0.0)]
    try:
        mc.seq.const.numBlocks = len(controls)
        for i, (nm, v) in enumerate(controls):
            mc.seq.const[i].par.name = nm
            mc.seq.const[i].par.value = v
    except Exception:
        for i, (nm, v) in enumerate(controls):
            setp(mc, **{f"name{i}": nm, f"value{i}": v})
    if mc["show_sonar"] is None:
        warn("main_controls: channels were not created")
    CNAMES = [c.name for c in mc.chans()]
    say(f"main_controls: {len(CNAMES)} channels")

    # ---------------------------------------------------------------- SONAR_RENDERER
    sr = fresh(rv, baseCOMP, "SONAR_RENDERER")
    srm = text_dat(rv, "sonar_renderer", os.path.join(TD_DIR, "sonar_renderer.py"))
    for nm, kind in (("sonar_positions", "points_slam"), ("sonar_positions_dr", "points_dr"), ("sonar_attributes", "attr")):
        script_top(sr, nm, f"    op('{ROOT_PATH}/sonar_renderer').module.cook(scriptOp, '{kind}')")
    n_side = 512
    sgeo = grid_geo(sr, "sonar_geometry", n_side, n_side, 6000.0, 6000.0, points=True)
    smat = make_mat(sr, "sonar_material", "sonar_points.vert", "sonar_points.frag")
    sgeo.par.material = smat
    ugeo = grid_geo(sr, "uncertainty_geometry_points", n_side, n_side, 6000.0, 6000.0, points=True)
    umat = make_mat(sr, "uncertainty_material", "sonar_points.vert", "uncertainty.frag")
    ugeo.par.material = umat
    CAM = lambda a: f"op('{P('CAMERA_SYSTEM/main_camera')}').par.t{a}.eval()"
    ylo, yhi = scene["floor_y"], scene["top_y"]
    for mat, frag_unc in ((smat, False), (umat, True)):
        mat.par.blending = True
        mat.par.srcblend = "one"
        mat.par.destblend = "one"
        mat.par.depthwriting = False
        set_sampler(mat, 0, "sPosDR", sr.op("sonar_positions_dr"))
        set_sampler(mat, 1, "sPosSL", sr.op("sonar_positions"))
        set_sampler(mat, 2, "sAttr", sr.op("sonar_attributes"))
        u = Uniforms(mat)
        u.f("uTime", TLS("time"))
        u.f("uMix", MC("slam_mix"))
        u.f("uPointSize", MC("point_size"))
        u.f("uRefDist", 600.0)
        u.f("uMaxRange", 30000.0)
        u.v3("uCamPos", [CAM("x"), CAM("y"), CAM("z")])
        u.f("uDensity", MC("density"))
        u.f("uHistoryS", 0.0)
        u.f("uExag", MC("exaggeration"))
        u.f("uExplodeMps", 0.0)
        u.f("uSonarRange", MC("max_range"))
        u.f("uDepthLo", ylo)
        u.f("uDepthHi", f"{ylo} + ({yhi} - ({ylo})) * {MC('depth_scale')}")
        u.f("uGain", f"{MC('gain')} * 0.5")
        u.f("uOpacity", MC("opacity"))
        if frag_unc:
            u.f("uSigLo", 0.3)
            u.f("uSigHi", 12.0)
        else:
            u.f("uColorMode", f"min({MC('color_mode')}, 4) - 1")
            u.f("uCinematic", MC("style_mode"))
            u.f("uClock", MC("clock"))
            u.f("uFreshS", 8.0)
            u.f("uDecay", 600.0)
    sgeo.par.render.expr = f"{MC('show_sonar')} > 0.5 and {MC('color_mode')} < 4.5"
    ugeo.par.render.expr = f"{MC('show_sonar')} > 0.5 and {MC('color_mode')} > 4.5"
    say("SONAR_RENDERER built")

    # ---------------------------------------------------------------- BATHYMETRY_RENDERER
    bt = fresh(rv, baseCOMP, "BATHYMETRY_RENDERER")
    for nm, kind in (("terrain_loader", "survey"), ("overview_loader", "overview"), ("recon_loader", "recon")):
        script_top(bt, nm, f"    op('{ROOT_PATH}/bathymetry_renderer').module.cook(scriptOp, '{kind}')")
    ex_s = bm.module.extent(rv, "survey")
    ex_o = bm.module.extent(rv, "overview")
    ex_r = bm.module.extent(rv, "recon")
    tgeo = grid_geo(bt, "terrain_geometry", 512, 512, ex_s[1] - ex_s[0], ex_s[3] - ex_s[2],
                    0.5 * (ex_s[0] + ex_s[1]), -0.5 * (ex_s[2] + ex_s[3]))
    ogeo = grid_geo(bt, "overview_geometry", 256, 256, ex_o[1] - ex_o[0], ex_o[3] - ex_o[2],
                    0.5 * (ex_o[0] + ex_o[1]), -0.5 * (ex_o[2] + ex_o[3]))
    rgeo = grid_geo(bt, "recon_geometry", 512, 512, ex_r[1] - ex_r[0], ex_r[3] - ex_r[2],
                    0.5 * (ex_r[0] + ex_r[1]), -0.5 * (ex_r[2] + ex_r[3]))
    mats = {}
    for nm, geo_, ext, src in (("terrain_material", tgeo, ex_s, 0), ("overview_material", ogeo, ex_o, 0),
                               ("recon_material", rgeo, ex_r, 1)):
        m = make_mat(bt, nm, "bathymetry.vert", "bathymetry.frag")
        geo_.par.material = m
        mats[nm] = m
        set_sampler(m, 0, "sHeight", bt.op("overview_loader") if nm == "overview_material" else bt.op("terrain_loader"))
        set_sampler(m, 1, "sRecon", bt.op("recon_loader"))
        u = Uniforms(m)
        u.v4("uExt", ext)
        u.f("uZRef", scene["z_ref"])
        u.f("uExag", MC("exaggeration"))
        u.f("uSource", src)
        u.f("uTime", TLS("time"))
        u.f("uColorMode", 0.0)
        u.f("uContour", f"25.0 if {MC('show_contours')} > 0.5 else 1e6")
        u.f("uGridStep", f"100.0 if {MC('show_grid')} > 0.5 else 1e7")
        u.f("uWire", MC("show_wire"))
        u.f("uDepthLo", scene["elev_min"])
        u.f("uDepthHi", f"{scene['elev_min']} + ({scene['elev_max']} - ({scene['elev_min']})) * {MC('depth_scale')}")
        u.v3("uCamPos", [CAM("x"), CAM("y"), CAM("z")])
        u.f("uCinematic", MC("style_mode"))
        u.f("uClock", MC("clock"))
        u.f("uFreshS", 10.0)
        u.f("uGain", MC("gain"))
        u.f("uOpacity", 1.0 if src else 0.55)
        u.f("uSigHi", 4.0)
        u.f("uBase", 1.0)
        u.f("uDuration", float(meta["duration_s"]))
    # terrain colour mode: depth for real terrain; reconstruction follows the global colour mode (depth / sigma / time)
    cm_recon = f"[0, 0, 3, 0, 1][int(min(max({MC('color_mode')}, 1), 5)) - 1]"
    set_vec(mats["recon_material"], 4, "uColorMode", 0.0)
    vec_expr(mats["recon_material"], 4, [cm_recon])
    for nm in ("terrain_material", "overview_material"):
        mats[nm].par.blending = True
        mats[nm].par.srcblend = "sa"
        mats[nm].par.destblend = "omsa"
        mats[nm].par.polygonoffset = True
        mats[nm].par.polygonoffsetfactor = 2.0
        mats[nm].par.polygonoffsetunits = 2.0
    rmat = mats["recon_material"]
    rmat.par.polygonoffset = True
    rmat.par.polygonoffsetfactor = -1.0
    rmat.par.polygonoffsetunits = -1.0
    tgeo.par.render.expr = f"{MC('show_terrain')} > 0.5 and {MC('terrain_view')} < 1.5"
    ogeo.par.render.expr = f"{MC('show_terrain')} > 0.5 and {MC('terrain_view')} > 1.5"
    rgeo.par.render.expr = f"{MC('show_recon')} > 0.5"
    ramp = fresh(bt, rampTOP, "depth_coloring")
    setp(ramp, outputresolution="custom", resolutionw=256, resolutionh=16)
    say("BATHYMETRY_RENDERER built")

    # ---------------------------------------------------------------- ACOUSTIC_BEAMS
    ab = fresh(rv, baseCOMP, "ACOUSTIC_BEAMS")
    text_dat(rv, "beam_renderer", os.path.join(TD_DIR, "beam_renderer.py"))
    bmat = make_mat(ab, "beam_material", "acoustic_beam.vert", "acoustic_beam.frag")
    bmat.par.blending = True
    bmat.par.srcblend = "one"
    bmat.par.destblend = "one"
    bmat.par.depthwriting = False
    bmat.par.cullface = "nocull" if "nocull" in list(bmat.par.cullface.menuNames) else bmat.par.cullface.eval()
    u = Uniforms(bmat)
    u.f("uExag", MC("exaggeration"))
    u.f("uGain", f"{MC('gain')} * 0.7")
    u.f("uOpacity", f"{MC('opacity')} * 0.55")
    u.f("uCinematic", MC("style_mode"))
    u.f("uClock", MC("clock"))
    u.f("uPulse", 0.35)
    bgeo, bsop = script_sop_geo(ab, "beam_geometry", (
        f"    base = op('{ROOT_PATH}')\n"
        f"    mc = base.op('UI_CONTROLS/main_controls')\n"
        f"    k = int(base.op('DATA_INPUT/timeline_state')['ping'].eval())\n"
        f"    op('{ROOT_PATH}/beam_renderer').module.build(scriptOp, base, k, mc['swath_deg'].eval())"), bmat)
    bgeo.par.render.expr = f"{MC('show_beams')} > 0.5"
    text_dat(ab, "beam_animation", text="# beam fan rebuild is driven by DATA_INPUT/frame_tick -> beam_geometry/sop.cook()\n")
    say("ACOUSTIC_BEAMS built")

    # ---------------------------------------------------------------- VEHICLE_NAVIGATION
    vn = fresh(rv, baseCOMP, "VEHICLE_NAVIGATION")
    vrm = text_dat(rv, "vehicle_renderer", os.path.join(TD_DIR, "vehicle_renderer.py"))
    vgeo = fresh(vn, geometryCOMP, "vehicle_geometry")
    vrm.module.build_vehicle_model(vgeo)
    vmat = fresh(vn, constantMAT, "vehicle_material")
    setp(vmat, colorr=0.1, colorg=1.0, colorb=1.0)
    vgeo.par.material = vmat
    vgeo.par.render.expr = f"{MC('show_vehicle')} > 0.5"
    ring = fresh(vn, geometryCOMP, "uncertainty_geometry")
    for c in list(ring.children):
        c.destroy()
    circ = ring.create(circleSOP, "ring")
    setp(circ, type="polygon", orient="zx" if "zx" in list(circ.par.orient.menuNames) else circ.par.orient.eval(),
         divs=96, radx=1.0, rady=1.0)
    circ.display = circ.render = True
    ring.par.material = line_mat(vn, "uncertainty_material", (1.0, 0.1, 0.9), 2.0, 0.9)
    ring.par.render.expr = f"{MC('show_nav_unc')} > 0.5"
    tr_top = {}
    for nm, kind, rgb, expr in (("dr", "traj_dr", (1.0, 0.35, 0.0), f"{MC('show_dr')} > 0.5 and {MC('show_traj')} > 0.5"),
                                ("slam", "traj_slam", (0.0, 1.0, 1.0), f"{MC('show_traj')} > 0.5"),
                                ("truth", "traj_truth", (0.45, 1.0, 0.0), f"{MC('show_truth_traj')} > 0.5")):
        t = script_top(vn, f"traj_{nm}", f"    op('{ROOT_PATH}/sonar_renderer').module.cook(scriptOp, '{kind}')")
        m = make_mat(vn, f"trajectory_material_{nm}", "trajectory.vert", "trajectory.frag")
        m.par.blending = True
        m.par.srcblend = "one"
        m.par.destblend = "one"
        m.par.depthwriting = False
        set_sampler(m, 0, "sPath", t)
        u = Uniforms(m)
        u.f("uTime", TLS("time"))
        u.f("uExag", MC("exaggeration"))
        u.f("uPointSize", 4.0)
        u.f("uRefDist", 600.0)
        u.v3("uCamPos", [CAM("x"), CAM("y"), CAM("z")])
        u.f("uTrailS", 0.0)
        u.f("uClock", MC("clock"))
        u.f("uPulseSpeed", 0.4)
        u.f("uPulseCount", 40.0)
        u.f("uCinematic", MC("style_mode"))
        u.v3("uColor", list(rgb))
        u.f("uGain", MC("gain"))
        u.f("uOpacity", 1.0)
        g_ = grid_geo(vn, f"trajectory_{nm}", 160, 160, 6000.0, 6000.0, points=True)
        g_.par.material = m
        g_.par.render.expr = expr
        tr_top[nm] = g_
    say("VEHICLE_NAVIGATION built")

    # ---------------------------------------------------------------- UNDERWATER_SLAM
    us = fresh(rv, baseCOMP, "UNDERWATER_SLAM")
    lgeo, lsop = script_sop_geo(us, "loop_closures",
                                f"    op('{ROOT_PATH}/vehicle_renderer').module.loops_into_sop(scriptOp, op('{ROOT_PATH}'))",
                                line_mat(us, "loop_material", (0.75, 1.0, 0.0), 1.5, 0.65))
    lgeo.par.render.expr = f"{MC('show_loops')} > 0.5"
    pgeo, psop = script_sop_geo(us, "pose_graph",
                                f"    op('{ROOT_PATH}/vehicle_renderer').module.version_path_into_sop(scriptOp, op('{ROOT_PATH}'))",
                                line_mat(us, "pose_graph_material", (1.0, 0.0, 0.8), 3.0, 1.0))
    pgeo.par.render.expr = f"{MC('show_versions')} > 0.5"
    vers = np.load(os.path.join(DATA_DIR, "versions.npz"))
    table(us, "map_versions", [["version", "keyframes", "loops", "t_end_s"]] +
          [[i + 1, int(vers["K"][i]), int(vers["n_loops"][i]), f"{float(vers['t_end'][i]):.0f}"] for i in range(len(vers["K"]))])
    say("UNDERWATER_SLAM built")

    # ---------------------------------------------------------------- OBJECT_PERCEPTION
    op_ = fresh(rv, baseCOMP, "OBJECT_PERCEPTION")
    ob = json.load(open(os.path.join(DATA_DIR, "objects.json")))
    table(op_, "object_candidates", [["id", "t_s", "E_m", "N_m", "height_m", "confidence", "status"]] +
          [[c["id"], f"{c['timestamp_s']:.0f}", f"{c['position_enu_m'][0]:.0f}", f"{c['position_enu_m'][1]:.0f}",
            f"{c['height_above_background_m']:.1f}", f"{c['confidence']:.2f}", c["classification_status"]] for c in ob["candidates"]])
    ogm, _ = script_sop_geo(op_, "object_geometry",
                            f"    op('{ROOT_PATH}/vehicle_renderer').module.objects_into_sop(scriptOp, op('{ROOT_PATH}'), False)",
                            line_mat(op_, "object_material", (0.55, 1.0, 0.0), 2.5, 1.0))
    ogm.par.render.expr = f"{MC('show_objects')} > 0.5"
    tgm, _ = script_sop_geo(op_, "truth_object_geometry",
                            f"    op('{ROOT_PATH}/vehicle_renderer').module.objects_into_sop(scriptOp, op('{ROOT_PATH}'), True)",
                            line_mat(op_, "truth_object_material", (1.0, 1.0, 1.0), 1.5, 0.9))
    tgm.par.render.expr = f"{MC('show_truth_objects')} > 0.5"
    lbl = fresh(op_, textTOP, "object_labels")
    setp(lbl, text=f"{len(ob['candidates'])} object candidates (unclassified)  |  {len(ob['inserted_synthetic'])} inserted "
                   "synthetic objects (evaluation)", fontsizex=14, alignx="left", aligny="top", bgalpha=0.0,
         fontcolorr=0.55, fontcolorg=1.0, fontcolorb=0.0, outputresolution="custom", resolutionw=1280, resolutionh=720)
    say("OBJECT_PERCEPTION built")

    # ---------------------------------------------------------------- TEMPORAL_4D
    t4 = fresh(rv, baseCOMP, "TEMPORAL_4D")
    trm = text_dat(rv, "temporal_renderer", os.path.join(TD_DIR, "temporal_renderer.py"))
    text_dat(t4, "sonar_history", text="# sonar history = temporal_renderer.visible_scan_range / shader window (world-space)\n")
    tmat = make_mat(t4, "temporal_material", "temporal_points.vert", "temporal_points.frag")
    tmat.par.blending = True
    tmat.par.srcblend = "one"
    tmat.par.destblend = "one"
    tmat.par.depthwriting = False
    set_sampler(tmat, 0, "sPosDR", sr.op("sonar_positions_dr"))
    set_sampler(tmat, 1, "sPosSL", sr.op("sonar_positions"))
    set_sampler(tmat, 2, "sAttr", sr.op("sonar_attributes"))
    TR = lambda key: (f"op('{ROOT_PATH}/temporal_renderer').module.uniforms({MC('temporal_mode')}, "
                      f"op('{ROOT_PATH}/temporal_renderer').module.nearest_history({MC('scan_history')}), "
                      f"{MC('temporal_decay')}, {MC('explode_mps')})['{key}']")
    u = Uniforms(tmat)
    u.f("uTime", TLS("time"))
    u.f("uMix", MC("slam_mix"))
    u.f("uPointSize", f"{MC('point_size')} * 0.9")
    u.f("uRefDist", 600.0)
    u.f("uMaxRange", 30000.0)
    u.v3("uCamPos", [CAM("x"), CAM("y"), CAM("z")])
    u.f("uDensity", MC("density"))
    u.f("uHistoryS", TR("uHistoryS"))
    u.f("uExag", MC("exaggeration"))
    u.f("uSonarRange", MC("max_range"))
    u.f("uExplodeMps", TR("uExplodeMps"))
    u.f("uDecay", TR("uDecay"))
    u.f("uLayerSeconds", TR("uLayerSeconds"))
    u.f("uGain", f"{MC('gain')} * 0.9")
    u.f("uOpacity", MC("opacity"))
    u.f("uCinematic", MC("style_mode"))
    u.f("uClock", MC("clock"))
    tpg = grid_geo(t4, "temporal_points", 512, 512, 6000.0, 6000.0, points=True)
    tpg.par.material = tmat
    tpg.par.render.expr = f"{MC('temporal_mode')} > 1.5"
    fresh(t4, constantCHOP, "scan_echoes")
    fresh(t4, constantCHOP, "time_explosion")
    text_dat(t4, "time_scrubber", text="# observation time lives in DATA_INPUT/timeline_state (timeline_controller module)\n")
    say("TEMPORAL_4D built")

    # ---------------------------------------------------------------- CAMERA_SYSTEM
    cs = fresh(rv, baseCOMP, "CAMERA_SYSTEM")
    ccm = text_dat(cs, "camera_controller", os.path.join(TD_DIR, "camera_controller.py"))
    cam = fresh(cs, cameraCOMP, "main_camera")
    setp(cam, near=2.0, far=cfg_cam_far)
    for nm in ("vehicle_pov", "chase_camera", "orbit_camera", "free_camera", "overhead_camera", "cinematic_camera"):
        c = fresh(cs, cameraCOMP, nm)
        setp(c, near=2.0, far=cfg_cam_far)
    kb = fresh(cs, keyboardinCHOP, "keyboard")
    kb.par.keys = "w a s d q e shift"
    ms = fresh(cs, mouseinCHOP, "mouse")
    for nm, v in (("lbuttonname", "lbutton"), ("wheel", "wheel"), ("wheelinc", 1.0)):
        setp(ms, **{nm: v})
    say("CAMERA_SYSTEM built")

    # ---------------------------------------------------------------- OUTPUT (render, atmosphere, bloom, composite)
    out = fresh(rv, baseCOMP, "OUTPUT")
    r = fresh(out, renderTOP, "render3d")
    geos = [tgeo, ogeo, rgeo, sgeo, ugeo, tpg, bgeo, vgeo, ring, lgeo, pgeo, ogm, tgm] + list(tr_top.values())
    setp(r, camera=cam, geometry=" ".join(g.path for g in geos), lights="", outputresolution="custom",
         resolutionw=meta.get("render_w", 1280), resolutionh=meta.get("render_h", 720),
         bgcolorr=0.004, bgcolorg=0.0, bgcolorb=0.02, bgcolora=1.0, format="rgba16float")
    atm = fresh(out, glslTOP, "underwater_atmosphere")
    atm_pix = text_dat(out, "underwater_atmosphere_pixel", os.path.join(SHADER_DIR, "underwater_fog.frag"))
    setp(atm, pixeldat=atm_pix, format="rgba16float")
    atm.inputConnectors[0].connect(r)
    UI = lambda fn: f"op('{P('UI_CONTROLS/ui_controller')}').module.post_params({MC('style_mode')}, {MC('show_bloom')}, " \
                    f"{MC('glow')}, {MC('show_atmosphere')})['{fn}']"
    for i, nm in enumerate(("uHaze", "uVignette", "uSparkle", "uScan", "uChroma")):
        set_vec(atm, i, nm, 0.0)
        vec_expr(atm, i, [UI(nm)])
    set_vec(atm, 5, "uClock", 0.0)
    vec_expr(atm, 5, [MC("clock")])
    bl = fresh(out, glslTOP, "bloom")
    bl_pix = text_dat(out, "bloom_pixel", os.path.join(SHADER_DIR, "glow.frag"))
    setp(bl, pixeldat=bl_pix, format="rgba16float")
    bl.inputConnectors[0].connect(atm)
    for i, nm in enumerate(("uGlow", "uRadius", "uThreshold")):
        set_vec(bl, i, nm, 0.0)
        vec_expr(bl, i, [UI(nm)])
    diag = fresh(out, textTOP, "diagnostics_text")
    setp(diag, fontsizex=13, alignx="left", aligny="top", bgalpha=0.0, fontcolorr=0.45, fontcolorg=1.0, fontcolorb=0.95,
         outputresolution="custom", resolutionw=1280, resolutionh=720, font="Consolas")
    diag.par.text.expr = (f"op('{P('UI_CONTROLS/ui_controller')}').module.diagnostics_text({TLS('time')}, {TLS('ping')}, "
                          f"{TLS('version')}, {MC('slam_mix')}, {MC('style_mode')}, {MC('color_mode')})")
    ttl = fresh(out, textTOP, "title_overlay")
    setp(ttl, fontsizex=30, alignx="left", aligny="bottom", bgalpha=0.0, fontcolorr=1.0, fontcolorg=0.2, fontcolorb=0.9,
         outputresolution="custom", resolutionw=1280, resolutionh=720, font="Consolas", text="")
    tl_ = fresh(out, levelTOP, "title_visibility")
    tl_.inputConnectors[0].connect(ttl)
    tl_.par.opacity.expr = MC("title_alpha")
    comp = fresh(out, compositeTOP, "final_composite")
    comp.par.operand = "over"
    comp.inputConnectors[0].connect(tl_)
    comp.inputConnectors[1].connect(diag)
    comp.inputConnectors[2].connect(bl)
    fd = fresh(out, levelTOP, "fade_to_black")
    fd.inputConnectors[0].connect(comp)
    fd.par.brightness1.expr = MC("fade")
    nul = fresh(out, nullTOP, "OUT")
    nul.inputConnectors[0].connect(fd)
    nul.viewer = True
    mo = fresh(out, moviefileoutTOP, "movie_output")
    try:
        mo.inputConnectors[0].connect(nul)
        mo.par.file = os.path.join(PROJECT_DIR, "output", "videos", "realtime_capture.mp4")
        mo.par.record = False
    except Exception as e:
        warn(f"movie_output: {e}")
    ow = fresh(out, windowCOMP, "output_window")
    setp(ow, winop=nul, winw=1280, winh=720)
    text_dat(out, "export_controller", os.path.join(TD_DIR, "export_controller.py"))
    say("OUTPUT built")

    # ---------------------------------------------------------------- UI panel (widgets are the source of truth)
    PW, PH = 900, 700
    panel = fresh(ui, containerCOMP, "panel")
    setp(panel, w=PW, h=PH, bgcolorr=0.02, bgcolorg=0.0, bgcolorb=0.06, bgalpha=1.0)
    W = {}
    cur = {"x": 12, "y": PH - 8}

    def place(o, x, y, w, h):
        o.par.x, o.par.y, o.par.w, o.par.h = x, y, w, h

    def label(text, x, y, w, h=18, size=13, rgb=(0.6, 0.7, 0.9), name=None):
        t = panel.create(textCOMP, name or "lbl")
        if name:
            t.name = name
        setp(t, text=text, fontsizex=size, fontcolorr=rgb[0], fontcolorg=rgb[1], fontcolorb=rgb[2], alignx="left")
        place(t, x, y, w, h)
        return t

    def button(key, text, x, y, w, kind, default=0, on=(1.0, 0.0, 0.8)):
        b = panel.create(buttonCOMP, key)
        b.name = key
        setp(b, buttontype=kind, label=text, bgcolorr=0.10, bgcolorg=0.05, bgcolorb=0.18,
             colorr=on[0], colorg=on[1], colorb=on[2])
        place(b, x, y, w, 24)
        if kind != "momentary":
            b.par.value0 = default
        W[key] = b
        return b

    def header(text, x, col_w):
        cur["y"] -= 20
        label(text, x, cur["y"], col_w, 16, 12, (0.0, 1.0, 0.9))

    def radio(prefix, items, default, x, col_w):
        cur["y"] -= 28
        wdt = (col_w - 4) / len(items)
        keys = []
        for i, it in enumerate(items):
            k = f"{prefix}{i + 1}"
            b = button(k, it, x + i * wdt, cur["y"], wdt - 4, "radiodown", 1 if i + 1 == default else 0)
            b.par.buttongroup = prefix
            keys.append(k)
        return keys

    def slider(key, text, lo, hi, dflt, x, col_w):
        cur["y"] -= 23
        label(text, x, cur["y"], 150, 20, 12, (0.78, 0.82, 0.95), name="l_" + key)
        sl = panel.create(sliderCOMP, "s_" + key)
        sl.name = "s_" + key
        place(sl, x + 152, cur["y"] + 1, col_w - 152 - 56, 18)
        setp(sl, valuerange0l=lo, valuerange0h=hi, value0=dflt)
        W["s_" + key] = sl
        vt = label("", x + col_w - 52, cur["y"], 52, 20, 12, (1.0, 0.85, 0.2), name="v_" + key)
        try:
            vt.par.text.expr = f"'%.2f' % op('{sl.path}').par.value0.eval()"
        except Exception as e:
            warn(f"value label {key}: {e}")

    COL = 430
    X0, X1 = 12, 12 + COL + 20
    label("DEEP OCEAN ROBOT VISION - SONAR 4D", X0, PH - 30, 600, 24, 19, (1.0, 0.1, 0.85), name="title")
    cur["y"] = PH - 34
    header("RENDER STYLE", X0, COL)
    style_keys = radio("r_style", ["SCIENTIFIC", "CINEMATIC"], 2, X0, COL)
    header("COLOUR MODE", X0, COL)
    color_keys = radio("r_color", ["INTENSITY", "DEPTH", "AGE", "SCAN BAND", "UNCERTAIN"], 2, X0, COL)
    header("TEMPORAL 4D", X0, COL)
    temporal_keys = radio("r_temporal", ["OFF", "SONAR ECHO", "SCAN LAYERS", "TIME EXPLOSION"], 1, X0, COL)
    header("CAMERA", X0, COL)
    camera_keys = radio("r_camera", ["VEHICLE", "CHASE", "ORBIT", "FREE", "OVERHEAD", "CINEMA"], 1, X0, COL)
    header("TERRAIN", X0, COL)
    terrain_keys = radio("r_terrain", ["SURVEY 3.5 m", "OVERVIEW 17.5 m"], 1, X0, COL)
    header("PLAYBACK", X0, COL)
    cur["y"] -= 28
    for i, (key, text) in enumerate([("b_play", "PLAY"), ("b_pause", "PAUSE"), ("b_reverse", "REVERSE"),
                                     ("b_step_back", "STEP -"), ("b_step_fwd", "STEP +"), ("b_reset", "RESET")]):
        button(key, text, X0 + i * 72, cur["y"], 68, "momentary")
    cur["y"] -= 28
    button("t_freeze", "FREEZE TIME", X0, cur["y"], 130, "toggledown", 0, (0.2, 0.9, 1.0))
    button("b_jump_scan", "JUMP TO SCAN", X0 + 140, cur["y"], 130, "momentary")
    button("b_jump_version", "JUMP TO MAP VER.", X0 + 280, cur["y"], 140, "momentary")
    slider("jump_scan", "scan index", 0, meta["n_pings"] - 1, 0, X0, COL)
    slider("jump_version", "map version index", 0, max(meta["n_versions"] - 1, 1), 0, X0, COL)
    slider("slam_mix", "DR (0) <-> SLAM (1)", 0.0, 1.0, 1.0, X0, COL)
    left_end = cur["y"]

    cur["y"] = PH - 34
    header("LAYERS", X1, COL)
    toggles = [("show_sonar", "SONAR"), ("show_beams", "BEAMS"), ("show_terrain", "REAL TERRAIN"), ("show_recon", "RECON"),
               ("show_wire", "WIRE MESH"), ("show_vehicle", "VEHICLE"), ("show_traj", "TRAJECTORY"), ("show_dr", "DR PATH"),
               ("show_truth_traj", "SIM TRUTH"), ("show_nav_unc", "NAV UNCERT."), ("show_loops", "LOOP CLOSE"),
               ("show_objects", "OBJECTS"), ("show_truth_objects", "SIM OBJECTS"), ("show_versions", "MAP VERSION"),
               ("show_contours", "CONTOURS"), ("show_grid", "LED GRID"), ("show_atmosphere", "ATMOSPHERE"),
               ("show_bloom", "BLOOM")]
    dflt_on = {c[0]: c[1] for c in controls}
    for i, (key, text) in enumerate(toggles):
        if i % 3 == 0:
            cur["y"] -= 28
        button("t_" + key, text, X1 + (i % 3) * 144, cur["y"], 140, "toggledown", int(dflt_on[key]), (0.2, 1.0, 0.2))
    header("CONTROLS", X1, COL)
    sliders = [("point_size", "point size", 1.0, 8.0, 3.0), ("opacity", "sonar opacity", 0.0, 1.0, 1.0),
               ("gain", "acoustic gain", 0.1, 3.0, 1.0), ("max_range", "max sonar range (m)", 20.0, 260.0, 260.0),
               ("swath_deg", "beam fan shown (deg)", 10.0, 130.0, 130.0), ("scan_history", "scan history (scans)", 5, 500, 50),
               ("temporal_decay", "temporal decay (s)", 5.0, 300.0, 40.0), ("exaggeration", "vert. exaggeration (visual)", 1.0, 8.0, 2.0),
               ("depth_scale", "depth colour scale", 0.3, 2.0, 1.0), ("camera_speed", "camera speed (m/s)", 20.0, 1500.0, 260.0),
               ("playback_speed", "playback speed (x)", 1.0, 600.0, 60.0), ("glow", "glow intensity", 0.0, 2.0, 0.9),
               ("density", "point density", 0.05, 1.0, 1.0),
               ("explode_mps", "time axis (m/s, artistic)", 0.05, 2.0, 0.5)]
    for key, text, lo, hi, d in sliders:
        slider(key, text, lo, hi, d, X1, COL)
    label("vertical exaggeration is a VISUALISATION setting; positions in the exported data are metric.",
          X1, min(cur["y"], left_end) - 22, COL, 16, 11, (1.0, 0.8, 0.2), name="note_exag")

    # bind main_controls channels to the widgets
    def bind(ch, expr):
        try:
            mc.seq.const[CNAMES.index(ch)].par.value.expr = expr
        except Exception as e:
            warn(f"bind {ch}: {e}")
    val = lambda key: f"op('{W[key].path}').par.value0.eval()"
    for key, _t in toggles:
        bind(key, f"float({val('t_' + key)})")
    for key in [s_[0] for s_ in sliders] + ["jump_scan", "jump_version", "slam_mix"]:
        bind(key, f"float({val('s_' + key)})")
    UIM = f"op('{P('UI_CONTROLS/ui_controller')}').module"
    grp = lambda keys: "[" + ", ".join(val(k) for k in keys) + "]"
    bind("style_mode", f"{UIM}.radio_index({grp(style_keys)}, 2) - 1")
    bind("color_mode", f"{UIM}.radio_index({grp(color_keys)}, 2)")
    bind("temporal_mode", f"{UIM}.radio_index({grp(temporal_keys)}, 1)")
    bind("camera_mode", f"{UIM}.radio_index({grp(camera_keys)}, 1)")
    bind("terrain_view", f"{UIM}.radio_index({grp(terrain_keys)}, 1)")

    PANEL_CB = (
        "BASE = '%s'\n"
        "def onOffToOn(panelValue):\n"
        "    n = panelValue.owner.name\n"
        "    base = op(BASE)\n"
        "    tl = base.op('DATA_INPUT/timeline_controller').module\n"
        "    mc = base.op('UI_CONTROLS/main_controls')\n"
        "    if n == 'b_play': tl.play()\n"
        "    elif n == 'b_pause': tl.pause()\n"
        "    elif n == 'b_reverse': tl.reverse()\n"
        "    elif n == 'b_step_back': tl.step(-1)\n"
        "    elif n == 'b_step_fwd': tl.step(1)\n"
        "    elif n == 'b_reset': tl.reset()\n"
        "    elif n == 'b_jump_scan': tl.jump_to_scan(int(mc['jump_scan'].eval()))\n"
        "    elif n == 'b_jump_version': tl.jump_to_version(int(mc['jump_version'].eval()))\n"
        "    return\n" % ROOT_PATH)
    pex = fresh(ui, panelexecuteDAT, "panel_events")
    pex.text = PANEL_CB
    try:
        pex.par.panels = " ".join(W[k].path for k in W if k.startswith("b_"))
        pex.par.offtoon = True
    except Exception as e:
        warn(f"panel_events: {e}")
    pst = fresh(ui, panelexecuteDAT, "panel_state_events")
    pst.text = ("def onValueChange(panelValue, prev):\n"
                f"    op('{ROOT_PATH}/DATA_INPUT/timeline_controller').module.freeze(bool(panelValue.val))\n    return\n")
    try:
        pst.par.panels = W["t_freeze"].path
        pst.par.panelvalue = "state"
        pst.par.valuechange = True
    except Exception as e:
        warn(f"panel_state_events: {e}")
    win = fresh(ui, windowCOMP, "control_window")
    setp(win, winop=panel, winw=PW, winh=PH)
    prev = fresh(ui, opviewerTOP, "panel_preview")
    setp(prev, opviewer=panel, outputresolution="custom", resolutionw=PW, resolutionh=PH)
    for grp_name, chans in (("sonar_controls", ["point_size", "opacity", "gain", "max_range", "swath_deg", "density"]),
                            ("terrain_controls", ["exaggeration", "depth_scale", "terrain_view", "show_contours", "show_grid"]),
                            ("temporal_controls", ["scan_history", "temporal_decay", "temporal_mode", "playback_speed", "slam_mix"]),
                            ("camera_controls", ["camera_mode", "camera_speed", "cam_fov"])):
        sel = fresh(ui, selectCHOP, grp_name)
        setp(sel, chops=mc.path, channames=" ".join(chans))
    fresh(ui, nullCHOP, "diagnostics").inputConnectors[0].connect(tls)
    say(f"UI panel built: {len(W)} widgets")

    # ---------------------------------------------------------------- frame hook
    ex = fresh(di, executeDAT, "frame_tick")
    ex.text = (
        "def onFrameStart(frame):\n"
        f"    base = op('{ROOT_PATH}')\n"
        "    try:\n"
        "        base.op('DATA_INPUT/timeline_controller').module.tick()\n"
        "        mc = base.op('UI_CONTROLS/main_controls')\n"
        "        mc.par.value%d = absTime.seconds\n"
        "        base.op('CAMERA_SYSTEM/camera_controller').module.update()\n"
        f"        op('{ROOT_PATH}/vehicle_renderer').module.apply_pose(base)\n"
        "        base.op('ACOUSTIC_BEAMS/beam_geometry/sop').cook(force=True)\n"
        "        base.op('UNDERWATER_SLAM/loop_closures/sop').cook(force=True)\n"
        "        base.op('UNDERWATER_SLAM/pose_graph/sop').cook(force=True)\n"
        "    except Exception as e:\n"
        "        print('frame_tick error:', e)\n"
        "    return\n" % CNAMES.index("clock"))
    ex.par.framestart = True
    say(f"BUILD OK ({len(WARN)} warnings)")
    return rv


rv_ = build()
if SAVE:
    _toe = os.path.join(TD_DIR, "Deep_Ocean_Robot_Vision_4D.toe")
    project.save(_toe)                    # same folder as base.tox / modules, so the MCP bridge routes keep working
    print("saved", _toe)
