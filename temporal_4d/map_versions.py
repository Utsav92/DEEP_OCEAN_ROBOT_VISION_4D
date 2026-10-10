"""Map versions: incremental SLAM solutions (each = keyframe poses valid when version K keyframes had been seen)."""
import numpy as np


class MapVersions:
    def __init__(self, K, t_end, n_loops, T, kf_T_dr, kf_t):
        self.K, self.t_end, self.n_loops = np.asarray(K), np.asarray(t_end), np.asarray(n_loops)
        self.T = np.asarray(T)                 # (V, Kmax, 4, 4)
        self.kf_T_dr, self.kf_t = np.asarray(kf_T_dr), np.asarray(kf_t)

    @classmethod
    def from_slam_npz(cls, path):
        d = np.load(path)
        return cls(d["version_K"], d["version_t"], d["version_loops"], d["version_T"], d["kf_T_dr"], d["kf_t"])

    def __len__(self):
        return len(self.K)

    def version_at_time(self, t):
        return int(np.clip(np.searchsorted(self.t_end, t, side="right") - 1, 0, len(self.K) - 1))

    def trajectory(self, v):
        """Keyframe positions (K_v,3) and times for version v; later keyframes (not yet seen) are excluded."""
        k = int(self.K[v])
        return self.T[v, :k, :3, 3], self.kf_t[:k]

    def correction_magnitude(self, v):
        k = int(self.K[v])
        return np.linalg.norm(self.T[v, :k, :3, 3] - self.kf_T_dr[:k, :3, 3], axis=1)
