"""Data bridge: file-based playback (default) and a localhost TCP packet stream, plus the frame-packet protocol.

Pure python + numpy: importable inside TouchDesigner (Text DAT module) and in the test-suite.

File playback
    DatasetStore(dir) memory-maps the .npy exports of scripts/build_map.py (see temporal_4d/temporal_export.py).
Packet stream (protocol version 1)
    One JSON header line per ping:
      {"protocol":1, "seq":int, "timestamp_s":float, "frame":"TD(x=E,y=U-zref,z=-N)", "sensor_pose":[16 floats],
       "vehicle_pose":[16 floats], "pointcloud_ref":{"file":"map_points_slam.npy","slots":[a,b]},
       "attribute_ref":{"file":"map_attr.npy","slots":[a,b]}, "map_version":int, "validity":{"pose":bool,"points":bool,...}}
    PacketReceiver tolerates missing, duplicated, delayed and out-of-order packets (reorder window + gap accounting).
"""
import json
import os
import socket
import threading
from dataclasses import dataclass, field, asdict

import numpy as np

PROTOCOL_VERSION = 1
FRAME_ID = "TD(x=E,y=U-zref,z=-N)"


@dataclass
class FramePacket:
    seq: int
    timestamp_s: float
    sensor_pose: list
    vehicle_pose: list
    pointcloud_ref: dict
    attribute_ref: dict
    map_version: int = 0
    validity: dict = field(default_factory=dict)
    protocol: int = PROTOCOL_VERSION
    frame: str = FRAME_ID

    def to_json(self):
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        if d.get("protocol") != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol {d.get('protocol')}")
        for k in ("seq", "timestamp_s", "sensor_pose", "vehicle_pose", "pointcloud_ref", "attribute_ref"):
            if k not in d:
                raise ValueError(f"packet missing field {k}")
        return cls(**d)


class DatasetStore:
    """Memory-mapped access to the exported arrays + metadata."""

    def __init__(self, directory):
        self.dir = directory
        self.meta = json.load(open(os.path.join(directory, "td_meta.json")))
        self._cache = {}

    def array(self, name):
        if name not in self._cache:
            path = os.path.join(self.dir, name if name.endswith(".npy") else name + ".npy")
            # exports are small (<= ~20 MB): load fully so no file stays memory-mapped (Windows cannot overwrite those)
            big = os.path.getsize(path) > 64 * 1024 * 1024
            self._cache[name] = np.load(path, mmap_mode="r" if big else None)
        return self._cache[name]

    def json(self, name):
        return json.load(open(os.path.join(self.dir, name)))

    @property
    def ping_t(self):
        return np.asarray(self.array("ping_t"))

    @property
    def n_pings(self):
        return len(self.ping_t)

    def packet(self, k, slots_per_ping=None):
        Tv = np.asarray(self.array("vehicle_T_slam")[k]).ravel().tolist()
        vp = {"pose": True}
        return FramePacket(seq=int(k), timestamp_s=float(self.ping_t[k]), sensor_pose=Tv, vehicle_pose=Tv,
                           pointcloud_ref={"file": "map_points_slam.npy", "slots": slots_per_ping or [0, 0]},
                           attribute_ref={"file": "map_attr.npy", "slots": slots_per_ping or [0, 0]},
                           map_version=int(np.searchsorted(self.versions_t_end(), self.ping_t[k], side="right") - 1),
                           validity=vp)

    def versions_t_end(self):
        return np.asarray(np.load(os.path.join(self.dir, "versions.npz"))["t_end"])


class PacketReceiver:
    """Reorders packets by sequence id within a bounded window and reports gaps.

    push(packet) -> list of packets that became deliverable in order. A packet more than `window` positions behind the
    newest is dropped as stale; a gap that the window has moved past is declared lost (counted, not waited on forever)."""

    def __init__(self, window=8):
        self.window = int(window)
        self.next_seq = None
        self.buf = {}
        self.stats = {"delivered": 0, "late_dropped": 0, "duplicates": 0, "lost": 0, "reordered": 0}

    def push(self, p):
        out = []
        if self.next_seq is None:
            self.next_seq = p.seq
        if p.seq < self.next_seq:
            self.stats["late_dropped"] += 1
            return out
        if p.seq in self.buf:
            self.stats["duplicates"] += 1
            return out
        if p.seq != self.next_seq:
            self.stats["reordered"] += 1
        self.buf[p.seq] = p
        while True:
            while self.next_seq in self.buf:
                out.append(self.buf.pop(self.next_seq))
                self.next_seq += 1
                self.stats["delivered"] += 1
            if self.buf and max(self.buf) - self.next_seq >= self.window:
                lost_to = min(self.buf)                 # give up on the missing ones
                self.stats["lost"] += lost_to - self.next_seq
                self.next_seq = lost_to
                continue
            break
        return out

    def flush(self):
        out = []
        for s in sorted(self.buf):
            if s > self.next_seq:
                self.stats["lost"] += s - self.next_seq
            out.append(self.buf[s])
            self.next_seq = s + 1
            self.stats["delivered"] += 1
        self.buf.clear()
        return out


def serve(store, host="127.0.0.1", port=9871, rate_hz=5.0, start=0, stop=None, scramble=0, drop_every=0,
          stop_event=None, ready=None):
    """Stream packets to ONE client over TCP (JSON lines). `scramble` > 0 locally shuffles that many neighbours and
    `drop_every` > 0 omits every n-th packet - used to exercise the receiver. Blocking; run in a thread."""
    import random
    import time
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(30.0)
    if ready is not None:
        ready.set()
    conn, _ = srv.accept()
    stop = store.n_pings if stop is None else stop
    ids = list(range(start, stop))
    if drop_every:
        ids = [i for i in ids if (i - start) % drop_every != drop_every - 1]
    if scramble:
        rng = random.Random(1)
        for a in range(0, len(ids), scramble):
            seg = ids[a:a + scramble]
            rng.shuffle(seg)
            ids[a:a + scramble] = seg
    with conn:
        for i in ids:
            if stop_event is not None and stop_event.is_set():
                break
            conn.sendall((store.packet(i).to_json() + "\n").encode())
            if rate_hz:
                time.sleep(1.0 / rate_hz)
    srv.close()


class SocketClient(threading.Thread):
    """Background reader; thread-safe queue of in-order packets. Poll `take()` from TouchDesigner's main thread."""

    def __init__(self, host="127.0.0.1", port=9871, window=8):
        super().__init__(daemon=True)
        self.addr, self.rx, self.q, self.lock = (host, port), PacketReceiver(window), [], threading.Lock()
        self.error = None
        self.closed = False

    def run(self):
        try:
            s = socket.create_connection(self.addr, timeout=10)
            f = s.makefile("r")
            for line in f:
                try:
                    pk = FramePacket.from_json(line)
                except Exception as e:                       # malformed packet: skip, never crash the renderer
                    self.error = str(e)
                    continue
                with self.lock:
                    self.q += self.rx.push(pk)
            with self.lock:
                self.q += self.rx.flush()
        except Exception as e:
            self.error = str(e)
        self.closed = True

    def take(self):
        with self.lock:
            out, self.q = self.q, []
        return out


# ---------------------------------------------------------------------------------------------- pose helpers (numpy only)
def _mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s_ = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s_, (R[0, 2] - R[2, 0]) / s_, (R[1, 0] - R[0, 1]) / s_, 0.25 * s_]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s_ = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s_, (R[0, 1] + R[1, 0]) / s_, (R[0, 2] + R[2, 0]) / s_, (R[2, 1] - R[1, 2]) / s_]
    elif R[1, 1] > R[2, 2]:
        s_ = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s_, 0.25 * s_, (R[1, 2] + R[2, 1]) / s_, (R[0, 2] - R[2, 0]) / s_]
    else:
        s_ = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s_, (R[1, 2] + R[2, 1]) / s_, 0.25 * s_, (R[1, 0] - R[0, 1]) / s_]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def lerp_pose(T0, T1, a):
    q0, q1 = _mat_to_quat(T0[:3, :3]), _mat_to_quat(T1[:3, :3])
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(min(d, 1.0))
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    T = np.eye(4)
    T[:3, :3] = _quat_to_mat(q)
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    return T


def pose_at(store, t, mix=1.0, which=None):
    """Vehicle pose (TD axes, display metres, y = U - z_ref) at observation time t (seconds since first ping),
    blending dead-reckoned (mix 0) and SLAM-corrected (mix 1) poses."""
    pt = store.ping_t
    k = int(np.clip(np.searchsorted(pt, t, side="right") - 1, 0, len(pt) - 2))
    a = float(np.clip((t - pt[k]) / max(pt[k + 1] - pt[k], 1e-9), 0.0, 1.0))
    Ts = lerp_pose(np.asarray(store.array("vehicle_T_slam")[k]), np.asarray(store.array("vehicle_T_slam")[k + 1]), a)
    if mix >= 0.999:
        return Ts
    Td = lerp_pose(np.asarray(store.array("vehicle_T_dr")[k]), np.asarray(store.array("vehicle_T_dr")[k + 1]), a)
    return lerp_pose(Td, Ts, mix)


_stores = {}


def get_store(base):
    """TouchDesigner helper: DatasetStore for the directory kept in the component storage key 'data_dir'. Module
    globals can be wiped when a file-synced Text DAT reloads, so the store is rebuilt lazily."""
    d = base.fetch("data_dir")
    if d not in _stores:
        _stores[d] = DatasetStore(d)
    return _stores[d]
