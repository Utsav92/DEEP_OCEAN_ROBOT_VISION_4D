"""Map reintegration: when optimisation changes historical keyframe poses, re-place every ping and sounding with the
corrected poses. A world-frame correction C_k = T_opt_k T_dr_k^-1 is interpolated along the path between keyframes and
applied rigidly to the dead-reckoned world points (p_new = C p_old), so the map is rebuilt from corrected poses and the
raw measurements are never altered."""
import numpy as np
from scipy.spatial.transform import Rotation

from .pose_graph import inv_batch


def keyframe_corrections(T_dr, T_opt):
    return T_opt @ inv_batch(T_dr)


def ping_corrections(kf_s, C, ping_s):
    """Interpolate corrections (K,4,4) to every ping by path length. Clamped outside the keyframe range."""
    K = len(kf_s)
    a = np.clip(np.searchsorted(kf_s, ping_s, side="right") - 1, 0, K - 2)
    ds = np.maximum(kf_s[a + 1] - kf_s[a], 1e-9)
    al = np.clip((ping_s - kf_s[a]) / ds, 0.0, 1.0)
    Ra, Rb = Rotation.from_matrix(C[a, :3, :3]), Rotation.from_matrix(C[a + 1, :3, :3])
    rv = (Ra.inv() * Rb).as_rotvec()
    R = (Ra * Rotation.from_rotvec(al[:, None] * rv)).as_matrix()
    t = (1 - al)[:, None] * C[a, :3, 3] + al[:, None] * C[a + 1, :3, 3]
    out = np.tile(np.eye(4), (len(ping_s), 1, 1))
    out[:, :3, :3], out[:, :3, 3] = R, t
    return out


def apply_to_points(xyz, ping_idx, Cp):
    R, t = Cp[ping_idx, :3, :3], Cp[ping_idx, :3, 3]
    return np.einsum("nij,nj->ni", R, xyz) + t


def correct_poses(T_dr_pings, Cp):
    return Cp @ T_dr_pings
