"""Bounded, timestamped history of sonar scans stored in WORLD coordinates.

A 'scan' is one ping's valid soundings. History is a function of the (independently controllable) observation time,
never of the camera, and it is exact under scrubbing/reverse: the visible scans for time t are the n most recent pings
with ping_t <= t, found by binary search over the immutable arrays.
"""
import numpy as np

HISTORY_SIZES = (5, 10, 20, 50, 100, 500)


class SonarHistory:
    def __init__(self, ping_t, ping_offsets, xyz, intensity=None, max_scans=500):
        """ping_t (P,) sorted; ping_offsets (P+1,) sounding offsets per ping; xyz (M,3) world (sorted by ping)."""
        if np.any(np.diff(ping_t) <= 0):
            raise ValueError("ping times must be strictly increasing")
        self.ping_t = np.asarray(ping_t, dtype=np.float64)
        self.off = np.asarray(ping_offsets, dtype=np.int64)
        self.xyz = xyz
        self.intensity = intensity
        self.max_scans = int(max_scans)

    def __len__(self):
        return len(self.ping_t)

    def latest_index(self, t):
        """Index of the newest ping with ping_t <= t, or -1 before the first ping."""
        return int(np.searchsorted(self.ping_t, t, side="right") - 1)

    def window(self, t, n):
        """(first_ping, last_ping_inclusive) of the n most recent scans at time t (bounded by max_scans)."""
        n = min(int(n), self.max_scans)
        k = self.latest_index(t)
        if k < 0:
            return 0, -1
        return max(0, k - n + 1), k

    def points(self, t, n):
        a, b = self.window(t, n)
        if b < a:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
        s, e = self.off[a], self.off[b + 1]
        ping_of = np.repeat(np.arange(a, b + 1), np.diff(self.off[a:b + 2]))
        return self.xyz[s:e], t - self.ping_t[ping_of], ping_of

    def ages(self, t, n):
        a, b = self.window(t, n)
        return t - self.ping_t[a:b + 1] if b >= a else np.zeros(0)
