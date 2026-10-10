"""Timestamped SE(3) track with interpolation (linear position, SLERP rotation) and explicit gap/extrapolation flags."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PoseTrack:
    def __init__(self, t, positions, rotations):
        """t (N,) strictly increasing; positions (N,3); rotations (N,3,3)."""
        self.t = np.asarray(t, dtype=np.float64)
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        self.p = np.asarray(positions, dtype=np.float64)
        self.R = np.asarray(rotations, dtype=np.float64)
        self._slerp = Slerp(self.t, Rotation.from_matrix(self.R))

    def at(self, tq, max_gap_s=None):
        """Interpolated poses at query times. Returns (T (M,4,4), status (M,)) with status 0 = interpolated,
        1 = clamped (outside the track), 2 = inside a gap longer than max_gap_s."""
        tq = np.atleast_1d(np.asarray(tq, dtype=np.float64))
        tc = np.clip(tq, self.t[0], self.t[-1])
        status = np.where(tq != tc, 1, 0).astype(np.int8)
        R = self._slerp(tc).as_matrix()
        p = np.stack([np.interp(tc, self.t, self.p[:, k]) for k in range(3)], -1)
        if max_gap_s is not None:
            i = np.clip(np.searchsorted(self.t, tc) , 1, len(self.t) - 1)
            gap = self.t[i] - self.t[i - 1]
            status = np.where((gap > max_gap_s) & (status == 0), 2, status).astype(np.int8)
        T = np.tile(np.eye(4), (len(tq), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T, status

    def __len__(self):
        return len(self.t)
