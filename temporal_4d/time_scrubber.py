"""Observation-time controller (play / pause / reverse / step / freeze / jump). Pure python, shared by the tests and by
the TouchDesigner timeline module. Time is the OBSERVATION time of the data; the camera is independent of it."""
import numpy as np


class TimeScrubber:
    def __init__(self, ping_t, version_t=None, speed=60.0):
        self.ping_t = np.asarray(ping_t, dtype=np.float64)
        self.t0, self.t1 = float(self.ping_t[0]), float(self.ping_t[-1])
        self.version_t = None if version_t is None else np.asarray(version_t, dtype=np.float64)
        self.t = self.t0
        self.playing = False
        self.direction = 1
        self.speed = float(speed)          # data seconds per wall second
        self.frozen = False
        self.loop = True

    # ---- transport
    def play(self):
        self.playing, self.direction = True, 1

    def reverse(self):
        self.playing, self.direction = True, -1

    def pause(self):
        self.playing = False

    def freeze(self, on=True):
        self.frozen = bool(on)

    def reset(self):
        self.t, self.playing, self.direction, self.frozen = self.t0, False, 1, False

    def tick(self, dt_wall):
        """Advance by a wall-clock step. Frozen time does not advance (the camera is unaffected)."""
        if not self.playing or self.frozen:
            return self.t
        t = self.t + self.direction * self.speed * dt_wall
        if t > self.t1 or t < self.t0:
            t = self.t0 + (t - self.t0) % (self.t1 - self.t0) if self.loop else float(np.clip(t, self.t0, self.t1))
            if not self.loop:
                self.playing = False
        self.t = t
        return t

    # ---- discrete moves
    def scan_index(self):
        return int(np.clip(np.searchsorted(self.ping_t, self.t, side="right") - 1, 0, len(self.ping_t) - 1))

    def step(self, n=1):
        self.playing = False
        k = int(np.clip(self.scan_index() + n, 0, len(self.ping_t) - 1))
        self.t = float(self.ping_t[k])
        return k

    def jump_to_scan(self, k):
        k = int(np.clip(k, 0, len(self.ping_t) - 1))
        self.t = float(self.ping_t[k])
        return k

    def jump_to_version(self, v):
        if self.version_t is None:
            raise ValueError("no map versions available")
        v = int(np.clip(v, 0, len(self.version_t) - 1))
        self.t = float(self.version_t[v])
        return v

    def set_time(self, t):
        self.t = float(np.clip(t, self.t0, self.t1))
        return self.t
