"""SE(3) pose graph: nodes (keyframe vehicle poses T_world_vehicle), relative-pose edges, and absolute priors
(depth, gravity direction/roll-pitch, weak heading). Residuals are vectorised; optimisation lives in optimization.py.

Node parametrisation for optimisation: T_i = T0_i * Exp(delta_i), delta = [translation(3), rotation vector(3)] in the
node's own frame, so the large absolute rotations of FRD->ENU never approach the rotation-vector singularity.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Edge:
    i: int
    j: int
    Z: np.ndarray            # measured T_i^-1 T_j
    sigma_t: float           # m
    sigma_r: float           # rad
    kind: str                # odometry | sonar_odometry | loop_closure
    t: float = 0.0           # time at which the edge became available (later node time)
    info: dict = field(default_factory=dict)


@dataclass
class PoseGraph:
    T0: np.ndarray                          # (n,4,4) initial node poses (dead reckoning)
    t: np.ndarray                           # (n,)
    edges: list = field(default_factory=list)
    depth_meas: np.ndarray = None           # (n,) vehicle depth m positive down
    gravity_meas: np.ndarray = None         # (n,2) first two components of R^T [0,0,-1]... see gravity_vector
    yaw_meas: np.ndarray = None             # (n,)
    sigma_depth: float = 0.15
    sigma_tilt: float = np.radians(0.3)
    sigma_yaw: float = np.radians(3.0)
    anchor: int = 0

    @property
    def n(self):
        return len(self.T0)

    def add(self, e):
        self.edges.append(e)


def gravity_vector(R):
    """Unit 'down' direction expressed in the body frame: R^T * [0,0,-1] (ENU z is up, so down is -z)."""
    return -R[..., 2, :]      # row 2 of R^T column... R^T e_z-negated: (R^T)[:, 2] = R[2, :]


def exp_batch(delta):
    R = Rotation.from_rotvec(delta[:, 3:]).as_matrix()
    T = np.tile(np.eye(4), (len(delta), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = delta[:, :3]
    return T


def poses_from_delta(T0, delta):
    return T0 @ exp_batch(delta.reshape(-1, 6))


def inv_batch(T):
    Ti = np.tile(np.eye(4), (len(T), 1, 1))
    R = np.swapaxes(T[:, :3, :3], 1, 2)
    Ti[:, :3, :3] = R
    Ti[:, :3, 3] = -np.einsum("nij,nj->ni", R, T[:, :3, 3])
    return Ti


def residuals(x, g, edges_arr):
    """Stacked weighted residuals for delta vector x (6n)."""
    n = g.n
    T = poses_from_delta(g.T0, x)
    out = []
    # relative edges
    if len(edges_arr["i"]):
        Ti, Tj = T[edges_arr["i"]], T[edges_arr["j"]]
        E = inv_batch(edges_arr["Z"]) @ inv_batch(Ti) @ Tj
        rt = E[:, :3, 3] / edges_arr["st"][:, None]
        rr = Rotation.from_matrix(E[:, :3, :3]).as_rotvec() / edges_arr["sr"][:, None]
        out += [rt.ravel(), rr.ravel()]
    # depth prior (z up: z = -depth)
    if g.depth_meas is not None:
        out.append((T[:, 2, 3] + g.depth_meas) / g.sigma_depth)
    # gravity / tilt prior: body-frame down vector, first two components compared to the measured
    if g.gravity_meas is not None:
        gv = gravity_vector(T[:, :3, :3])
        out.append(((gv[:, :2] - g.gravity_meas) / g.sigma_tilt).ravel())
    # weak absolute heading from the compass (bias-prone, so loosely weighted)
    if g.yaw_meas is not None:
        yaw = np.arctan2(_fwd_east(T), _fwd_north(T))
        d = np.arctan2(np.sin(yaw - g.yaw_meas), np.cos(yaw - g.yaw_meas))
        out.append(d / g.sigma_yaw)
    # gauge: anchor node pose
    a = g.anchor
    da = x[6 * a:6 * a + 6]
    out.append(da / np.array([0.05, 0.05, 0.05, 1e-3, 1e-3, 1e-3]))
    return np.concatenate(out)


def _fwd_east(T):
    return T[:, 0, 0]          # vehicle forward (x_body) east component: R[0, 0] (ENU east row, body x col)


def _fwd_north(T):
    return T[:, 1, 0]


def pack_edges(g, edges=None):
    es = g.edges if edges is None else edges
    if not es:
        z = np.zeros(0)
        return {"i": z.astype(int), "j": z.astype(int), "Z": np.zeros((0, 4, 4)), "st": z, "sr": z}
    return {"i": np.array([e.i for e in es], int), "j": np.array([e.j for e in es], int),
            "Z": np.stack([e.Z for e in es]), "st": np.array([e.sigma_t for e in es]),
            "sr": np.array([e.sigma_r for e in es])}
