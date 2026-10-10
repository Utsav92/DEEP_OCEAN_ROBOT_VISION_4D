"""Vehicle pose history for trajectory trails and orientation display (poses are world-space and immutable)."""
import numpy as np


class VehicleHistory:
    def __init__(self, t, T_world_vehicle):
        self.t = np.asarray(t, dtype=np.float64)
        self.T = np.asarray(T_world_vehicle, dtype=np.float64)

    def index(self, t):
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def trail(self, t, n):
        k = self.index(t)
        a = max(0, k - n + 1)
        return self.T[a:k + 1, :3, 3], self.t[a:k + 1]

    def pose_at(self, t):
        """Interpolated pose at an arbitrary time (clamped to the recorded interval)."""
        from preprocessing.coordinate_systems import interpolate_pose
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        k = int(np.clip(np.searchsorted(self.t, tc, side="right") - 1, 0, len(self.t) - 2))
        a = (tc - self.t[k]) / max(self.t[k + 1] - self.t[k], 1e-12)
        return interpolate_pose(self.T[k], self.T[k + 1], a)
