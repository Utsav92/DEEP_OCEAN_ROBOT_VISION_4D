"""Vehicle model, pose application, navigation-uncertainty ring, loop-closure lines, map-version path and object boxes.

The vehicle is drawn at VEHICLE_SCALE times its real size (a 3 m AUV is sub-pixel in a 3 km scene); this is a
visualisation setting, not a measurement."""
import numpy as np

BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
VEHICLE_SCALE = 14.0
_cache = {}


def pose_matrix(base):
    """Current T_td_vehicle (display metres, exaggerated vertical) from the timeline + SLAM mix."""
    mc = base.op("UI_CONTROLS/main_controls")
    t = float(base.op("DATA_INPUT/timeline_state")["time"].eval())
    br = base.op("DATA_INPUT/data_bridge").module
    T = br.pose_at(br.get_store(base), t, float(mc["slam_mix"].eval())).copy()
    T[1, 3] *= float(mc["exaggeration"].eval())
    return T


def apply_pose(base):
    import td
    T = pose_matrix(base)
    geo = base.op("VEHICLE_NAVIGATION/vehicle_geometry")
    m = td.tdu.Matrix(*[float(x) for x in T.T.flatten()])
    s, r, t = m.decompose()
    geo.par.tx, geo.par.ty, geo.par.tz = t
    geo.par.rx, geo.par.ry, geo.par.rz = r
    geo.par.sx = geo.par.sy = geo.par.sz = VEHICLE_SCALE
    ring = base.op("VEHICLE_NAVIGATION/uncertainty_geometry")
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    t_s = float(base.op("DATA_INPUT/timeline_state")["time"].eval())
    sig = float(np.interp(t_s, st.ping_t, np.asarray(st.array("ping_sigma_dr"))))
    ring.par.tx, ring.par.ty, ring.par.tz = T[0, 3], T[1, 3], T[2, 3]
    ring.par.sx = ring.par.sz = max(sig, 0.5)
    ring.par.sy = 1.0
    return sig


def build_vehicle_model(geo):
    """Torpedo hull + nose + tail fins + nav lights in body (FRD) axes, 4 m long, via SOPs under `geo`."""
    for c in list(geo.children):
        c.destroy()
    parts = []
    hull = geo.create(tubeSOP, "hull")
    hull.par.type = "poly"
    hull.par.rad1 = hull.par.rad2 = 0.32
    hull.par.height = 3.2
    hull.par.rows = 2
    hull.par.cols = 16
    try:
        hull.par.orient = "x"
    except Exception:
        pass
    parts.append(hull)
    nose = geo.create(sphereSOP, "nose")
    nose.par.type = "poly"
    nose.par.radx = nose.par.rady = nose.par.radz = 0.32
    nose.par.tx = 1.6
    parts.append(nose)
    for i, (ty, tz, sz) in enumerate(((0.0, 0.32, 0.0), (0.32, 0.0, 0.0), (-0.32, 0.0, 0.0))):
        fin = geo.create(boxSOP, "fin%d" % i)
        fin.par.sizex, fin.par.sizey, fin.par.sizez = 0.5, (0.04 if ty else 0.7), (0.7 if ty else 0.04)
        fin.par.tx, fin.par.ty, fin.par.tz = -1.45, ty * 1.6, tz * 1.6
        parts.append(fin)
    mg = geo.create(mergeSOP, "merge1")
    mg.par.sops = " ".join(p.path for p in parts)
    mg.display = True
    mg.render = True
    return mg


def loops_into_sop(scriptOp, base):
    """Script SOP callback: draw accepted loop closures whose creation time <= now as straight lines."""
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    if "loops" not in _cache:
        _cache["loops"] = st.json("loops.json")
    t = float(base.op("DATA_INPUT/timeline_state")["time"].eval())
    n = sum(1 for l in _cache["loops"] if l["t_s"] <= t)
    exag = float(base.op("UI_CONTROLS/main_controls")["exaggeration"].eval())
    key = (n, round(exag, 2))
    if _cache.get("loops_key") == key:
        return n
    _cache["loops_key"] = key
    scriptOp.clear()
    for l in _cache["loops"]:
        if l["t_s"] > t:
            continue
        poly = scriptOp.appendPoly(2, closed=False, addPoints=True)
        for k, p in enumerate((l["p_i"], l["p_j"])):
            pt = poly[k].point
            pt.x, pt.y, pt.z = p[0], p[1] * exag, p[2]
    return n


def version_path_into_sop(scriptOp, base):
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    if "versions" not in _cache:
        _cache["versions"] = np.load(st.dir + "/versions.npz")
    V = _cache["versions"]
    v = int(base.op("DATA_INPUT/timeline_state")["version"].eval())
    exag = float(base.op("UI_CONTROLS/main_controls")["exaggeration"].eval())
    key = (v, round(exag, 2))
    if _cache.get("ver_key") == key:
        return v
    _cache["ver_key"] = key
    scriptOp.clear()
    k = int(V["K"][v])
    P = np.asarray(V["traj"][v, :k], dtype=np.float64)
    if len(P) >= 2:
        poly = scriptOp.appendPoly(len(P), closed=False, addPoints=True)
        for i, p in enumerate(P):
            pt = poly[i].point
            pt.x, pt.y, pt.z = float(p[0]), float(p[1] * exag), float(p[2])
    return v


def _seabed_elev(base):
    """Callable (x, y) -> real-bathymetry elevation (nearest texel) from the exported survey terrain texture."""
    import json
    import os
    d = os.path.join(os.path.dirname(base.fetch("data_dir")), "terrain")
    z = np.load(os.path.join(d, "terrain_survey.npy"))
    x0, x1, y0, y1 = json.load(open(os.path.join(d, "terrain_survey.json")))["extent_xxyy"]

    def f(x, y):
        ix = int(np.clip(round((x - x0) / (x1 - x0) * (z.shape[1] - 1)), 0, z.shape[1] - 1))
        iy = int(np.clip(round((y - y0) / (y1 - y0) * (z.shape[0] - 1)), 0, z.shape[0] - 1))
        return float(z[iy, ix])
    return f


def objects_into_sop(scriptOp, base, truth=False):
    """Object candidates or the inserted SYNTHETIC truth objects as wire boxes standing on the seabed.
    Box geometry in TD display axes: x = East, y = (elevation - z_ref) * exaggeration, z = -North."""
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    ob = st.json("objects.json")
    zr = float(st.meta["z_ref_m"])
    exag = float(base.op("UI_CONTROLS/main_controls")["exaggeration"].eval())
    seabed = _seabed_elev(base)
    scriptOp.clear()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    items = []
    if truth:
        for o in ob["inserted_synthetic"]:
            items.append((o["x"], o["y"], seabed(o["x"], o["y"]), o["length"] + 6, o["width"] + 6, o["height"] + 6))
    else:
        for c in ob["candidates"]:
            if c["confidence"] < 0.3:                       # low-confidence candidates stay in the table, not the 3D view
                continue
            e = c["extent_m"]
            x, y = c["position_enu_m"][0], c["position_enu_m"][1]
            items.append((x, y, seabed(x, y), e[0] + 8, e[1] + 8, max(c["height_above_background_m"], 4.0) + 6))
    for (x, y, zb, L, W, H) in items:
        base_y = (zb - zr) * exag
        c = np.array([[-L, -W, 0], [L, -W, 0], [L, W, 0], [-L, W, 0], [-L, -W, H], [L, -W, H], [L, W, H], [-L, W, H]]) * 0.5
        c[4:, 2] = H
        for a, b in edges:
            poly = scriptOp.appendPoly(2, closed=False, addPoints=True)
            for k, idx in enumerate((a, b)):
                pt = poly[k].point
                pt.x, pt.y, pt.z = float(x + c[idx, 0]), float(base_y + c[idx, 2] * exag), float(-(y + c[idx, 1]))
    return len(items)
