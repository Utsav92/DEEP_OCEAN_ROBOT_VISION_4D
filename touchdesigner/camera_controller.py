"""Camera system (Text-DAT module in CAMERA_SYSTEM). One render camera `main_camera` is driven every frame by six pose
generators; a mode change blends position (lerp), rotation (slerp) and fov over `cam_blend_s` seconds.

  1 Vehicle POV     at the sonar head, looking forward and down along the vehicle heading
  2 Chase           behind and above the vehicle in its heading
  3 Orbit           slow orbit around the mapped seabed centre (or the vehicle when 'orbit_target' = vehicle)
  4 Free-fly        W/A/S/D move, Q/E vertical, mouse look (left button), wheel = speed, Shift = fast
  5 Overhead        straight down above the survey, north up
  6 Cinematic       smooth Catmull-Rom path through the reconstructed environment, driven by the DATA clock

Poses are 4x4 column-vector world matrices in TouchDesigner display axes (y already includes the vertical
exaggeration); a camera looks down its local -z. Everything above the glue line is pure numpy and unit-tested.
"""
import math

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def ease(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def look_at(eye, target, up=UP):
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    z = _norm(eye - target)
    up = np.asarray(up, dtype=np.float64)
    if abs(np.dot(_norm(up), z)) > 0.999:
        up = np.array([0.0, 0.0, -1.0]) if abs(z[1]) > 0.5 else UP
    x = _norm(np.cross(up, z))
    y = np.cross(z, x)
    W = np.eye(4)
    W[:3, 0], W[:3, 1], W[:3, 2], W[:3, 3] = x, y, z, eye
    return W


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def blend_pose(Wa, Wb, a):
    W = np.eye(4)
    W[:3, :3] = quat_to_mat(slerp(mat_to_quat(Wa[:3, :3]), mat_to_quat(Wb[:3, :3]), a))
    W[:3, 3] = (1 - a) * Wa[:3, 3] + a * Wb[:3, 3]
    return W


# ----------------------------------------------------------------------------------------------- pose generators
def vehicle_axes(T_td):
    """(position, forward, starboard, down) of the vehicle in display axes."""
    R = T_td[:3, :3]
    return T_td[:3, 3].copy(), _norm(R[:, 0]), _norm(R[:, 1]), _norm(R[:, 2])


def pose_vehicle_pov(T_td, tilt_deg=22.0):
    p, f, s, d = vehicle_axes(T_td)
    th = math.radians(tilt_deg)
    look = _norm(f * math.cos(th) + np.array([0.0, -1.0, 0.0]) * math.sin(th))
    return look_at(p + np.array([0.0, 4.0, 0.0]), p + look * 60.0)


def pose_chase(T_td, back=140.0, height=60.0, ahead=60.0):
    p, f, _, _ = vehicle_axes(T_td)
    fh = _norm(np.array([f[0], 0.0, f[2]]))
    return look_at(p - fh * back + np.array([0.0, height, 0.0]), p + fh * ahead)


def pose_orbit(center, t, radius=1700.0, height=700.0, period_s=90.0):
    az = 2 * math.pi * t / period_s
    c = np.asarray(center, dtype=np.float64)
    return look_at(c + np.array([radius * math.sin(az), height, radius * math.cos(az)]), c)


def pose_overhead(center, height=3200.0):
    c = np.asarray(center, dtype=np.float64)
    return look_at(c + np.array([0.0, height, 0.0]), c, up=np.array([0.0, 0.0, -1.0]))


def catmull_rom(P, u):
    """Closed-parameter Catmull-Rom through points P (n,3), u in [0,1] over the whole open path."""
    n = len(P)
    x = min(max(u, 0.0), 0.999999) * (n - 1)
    i = int(x)
    f = x - i
    p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[min(i + 1, n - 1)], P[min(i + 2, n - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def cinematic_path(extent_xxzz, y_floor, exag, y_top=None):
    """Waypoints (eye) and look-at targets sweeping the reconstructed environment: low fly-through, then pull-back.
    Eyes stay above the highest terrain point (y_top) so the camera is never under the seabed."""
    x0, x1, z0, z1 = extent_xxzz
    cx, cz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
    W, H = x1 - x0, z1 - z0
    top = (y_floor + 460.0) if y_top is None else y_top
    h = lambda v: v * exag
    eyes = np.array([
        [x0 + 0.12 * W, h(top + 150), z1 - 0.10 * H],
        [x0 + 0.30 * W, h(top + 70), z1 - 0.35 * H],
        [cx - 0.05 * W, h(top + 90), cz],
        [cx + 0.25 * W, h(top + 60), z0 + 0.35 * H],
        [x1 - 0.15 * W, h(top + 160), z0 + 0.12 * H],
        [x1 + 0.35 * W, h(top + 700), cz + 0.2 * H],
        [cx + 1.1 * W, h(top + 1500), cz + 1.0 * H]])
    tgts = np.array([
        [cx - 0.30 * W, h(y_floor + 120), cz + 0.30 * H],
        [cx - 0.10 * W, h(y_floor + 120), cz + 0.10 * H],
        [cx + 0.10 * W, h(y_floor + 120), cz - 0.10 * H],
        [cx + 0.10 * W, h(y_floor + 120), cz - 0.30 * H],
        [cx, h(y_floor + 100), cz],
        [cx, h(y_floor + 100), cz],
        [cx, h(y_floor + 100), cz]])
    return eyes, tgts


def pose_cinematic(eyes, tgts, u):
    return look_at(catmull_rom(eyes, u), catmull_rom(tgts, u))


class FreeCam:
    def __init__(self):
        self.pos, self.yaw, self.pitch = np.zeros(3), 0.0, 0.0

    def from_matrix(self, W):
        f = _norm(-np.asarray(W)[:3, 2])
        self.pos = np.array(W[:3, 3], dtype=np.float64)
        self.pitch = math.asin(max(-1.0, min(1.0, f[1])))
        self.yaw = math.atan2(-f[0], -f[2])

    def forward(self):
        cp = math.cos(self.pitch)
        return np.array([-math.sin(self.yaw) * cp, math.sin(self.pitch), -math.cos(self.yaw) * cp])

    def step(self, dt, fwd=0.0, strafe=0.0, vert=0.0, dyaw=0.0, dpitch=0.0, speed=200.0):
        self.yaw += dyaw
        self.pitch = max(-1.5, min(1.5, self.pitch + dpitch))
        f = self.forward()
        right = _norm(np.cross(f, UP))
        self.pos = self.pos + (f * fwd + right * strafe + UP * vert) * speed * dt

    def matrix(self):
        return look_at(self.pos, self.pos + self.forward())


# ============================================ TouchDesigner glue ====================================================
BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
_st = {"mode": None, "W": None, "fov": 60.0, "from_W": None, "from_fov": 60.0, "t0": 0.0, "last": None,
       "free": FreeCam(), "inject": None, "override": None, "mouse": None, "wheel": None, "speed_mul": 1.0}


def set_override(W, fov):
    _st["override"] = (np.asarray(W, dtype=np.float64), float(fov))


def clear_override():
    _st["override"] = None


def inject(fwd=0.0, strafe=0.0, vert=0.0, dyaw=0.0, dpitch=0.0):
    _st["inject"] = dict(fwd=fwd, strafe=strafe, vert=vert, dyaw=dyaw, dpitch=dpitch)


def _apply(cam, td, W, fov):
    m = td.tdu.Matrix(*[float(x) for x in W.T.flatten()])
    s, r, t = m.decompose()
    cam.par.tx, cam.par.ty, cam.par.tz = t
    cam.par.rx, cam.par.ry, cam.par.rz = r
    cam.par.fov = fov


def _free_input(base):
    if _st["inject"] is not None:
        d, _st["inject"] = _st["inject"], None
        return d
    d = dict(fwd=0.0, strafe=0.0, vert=0.0, dyaw=0.0, dpitch=0.0)
    kb, mouse = base.op("CAMERA_SYSTEM/keyboard"), base.op("CAMERA_SYSTEM/mouse")
    if kb is not None:
        def key(n):
            c = kb["k" + n]
            return float(c.eval()) if c is not None else 0.0
        d["fwd"], d["strafe"], d["vert"] = key("w") - key("s"), key("d") - key("a"), key("e") - key("q")
        d["fast"] = key("shift") > 0.5
    if mouse is not None and mouse["tx"] is not None:
        x, y = float(mouse["tx"].eval()), float(mouse["ty"].eval())
        down = mouse["lbutton"] is not None and float(mouse["lbutton"].eval()) > 0.5
        if down and _st["mouse"] is not None:
            d["dyaw"], d["dpitch"] = -(x - _st["mouse"][0]) * 3.0, (y - _st["mouse"][1]) * 3.0
        _st["mouse"] = (x, y) if down else None
        if mouse["wheel"] is not None:
            w = float(mouse["wheel"].eval())
            if _st["wheel"] is not None:
                _st["speed_mul"] = min(max(_st["speed_mul"] * 1.25 ** (w - _st["wheel"]), 0.1), 20.0)
            _st["wheel"] = w
    return d


def scene_info(base):
    return base.fetch("scene", None) or {}


def target_pose(base, mode, t_data, cam_u=None):
    """(W, fov) of a scripted camera mode (1 vehicle POV, 2 chase, 3 orbit, 5 overhead, 6 cinematic) at observation time
    t_data. cam_u (0..1) overrides the cinematic path parameter (used by the video exporter)."""
    mc = base.op("UI_CONTROLS/main_controls")
    g = lambda n: float(mc[n].eval())
    sc = scene_info(base)
    exag = g("exaggeration")
    br = base.op("DATA_INPUT/data_bridge").module
    T = br.pose_at(br.get_store(base), t_data, g("slam_mix")).copy()
    T[1, 3] *= exag                                              # display (exaggerated) vertical
    c = np.array(sc.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    c[1] *= exag
    fov = g("cam_fov")
    if mode == 1:
        return pose_vehicle_pov(T), fov
    if mode == 2:
        return pose_chase(T, g("chase_back"), g("chase_up") + 0.5 * exag * 20.0), fov
    if mode == 3:
        return pose_orbit(c, t_data, g("orbit_radius"), g("orbit_height") * exag * 0.5, g("orbit_period")), fov
    if mode == 5:
        return pose_overhead(c, g("overhead_height")), fov
    ext = sc.get("extent_xxzz", [-1200, 1200, -1700, 1700])
    eyes, tgts = cinematic_path(ext, sc.get("floor_y", 0.0), exag, sc.get("top_y"))
    if cam_u is None:
        dur = float(base.op("DATA_INPUT/timeline_state")["duration"].eval())
        cam_u = (t_data / max(dur, 1.0)) % 1.0
    return pose_cinematic(eyes, tgts, cam_u), fov


def update():
    """Once per frame: compute the target pose, blend mode changes, drive main_camera. Returns the active mode."""
    import time
    import td
    base = td.op(BASE)
    mc = base.op("UI_CONTROLS/main_controls")
    tl = base.op("DATA_INPUT/timeline_state")
    cam = base.op("CAMERA_SYSTEM/main_camera")
    g = lambda n: float(mc[n].eval())
    t_data = float(tl["time"].eval())
    mode = int(round(g("camera_mode")))
    now = time.perf_counter()
    dt = 0.0 if _st["last"] is None else min(now - _st["last"], 0.25)
    _st["last"] = now

    if _st["override"] is not None:
        W, fov = _st["override"]
        _st.update(W=W, fov=fov, from_W=None)
        _apply(cam, td, W, fov)
        return mode

    if mode != 4:
        W, fov = target_pose(base, mode, t_data)
    else:                                                        # 4 free-fly
        free = _st["free"]
        if _st["mode"] != 4:
            free.from_matrix(_st["W"] if _st["W"] is not None else target_pose(base, 2, t_data)[0])
        d = _free_input(base)
        free.step(dt, d["fwd"], d["strafe"], d["vert"], d["dyaw"], d["dpitch"],
                  g("camera_speed") * _st["speed_mul"] * (3.0 if d.get("fast") else 1.0))
        W, fov = free.matrix(), g("cam_fov")

    if _st["W"] is None:
        _st["mode"], _st["W"], _st["fov"] = mode, W, fov
    if mode != _st["mode"]:
        _st.update(mode=mode, from_W=_st["W"], from_fov=_st["fov"], t0=now)
    a = ease((now - _st["t0"]) / max(g("cam_blend_s"), 1e-3)) if _st["from_W"] is not None else 1.0
    if a >= 1.0:
        _st["from_W"] = None
    if _st["from_W"] is not None:
        W, fov = blend_pose(_st["from_W"], W, a), (1 - a) * _st["from_fov"] + a * fov
    _st["W"], _st["fov"] = W, fov
    _apply(cam, td, W, fov)
    return mode
