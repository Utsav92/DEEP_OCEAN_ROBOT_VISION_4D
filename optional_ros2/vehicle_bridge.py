"""ROS 2 vehicle-state aggregator (MODE 6, optional; NEVER RUN). Combines the sonar and navigation sinks into
temporally ordered (ping, pose) pairs using the same PoseTrack interpolation as the offline pipeline."""
import numpy as np


class VehicleAggregator:
    """Push navigation poses and sonar pings in any order; pings are released once nav covers their timestamp."""

    def __init__(self, max_lag_s=2.0):
        self.nav_t, self.nav_T, self.pings, self.max_lag = [], [], [], max_lag_s

    def push_pose(self, t, T):
        if self.nav_t and t <= self.nav_t[-1]:
            return False                                   # out-of-order / duplicate pose
        self.nav_t.append(float(t))
        self.nav_T.append(np.asarray(T))
        return True

    def push_ping(self, ping):
        self.pings.append(ping)

    def pop_ready(self):
        """Returns [(ping, T_enu_vehicle)] for pings whose time lies inside the received navigation interval."""
        from preprocessing.coordinate_systems import interpolate_pose
        out, keep = [], []
        for p in self.pings:
            t = p[0]
            if len(self.nav_t) >= 2 and self.nav_t[0] <= t <= self.nav_t[-1]:
                k = int(np.clip(np.searchsorted(self.nav_t, t) - 1, 0, len(self.nav_t) - 2))
                a = (t - self.nav_t[k]) / max(self.nav_t[k + 1] - self.nav_t[k], 1e-9)
                out.append((p, interpolate_pose(self.nav_T[k], self.nav_T[k + 1], a)))
            elif self.nav_t and t < self.nav_t[0] - self.max_lag:
                continue                                    # too old: dropped, not guessed
            else:
                keep.append(p)
        self.pings = keep
        return out
