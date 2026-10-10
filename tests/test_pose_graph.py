import numpy as np
import pytest

from preprocessing import coordinate_systems as cs
from slam import pose_graph as pg, optimization as opt, map_reintegration as mr


def _square_loop(n_side=10, step=10.0, drift=0.08):
    """Dead reckoning around a square returning to its start: a pure scale-style drift opens the loop."""
    pts, yaw = [np.zeros(3)], []
    for side in range(4):
        h = side * np.pi / 2
        d = np.array([np.sin(h), np.cos(h), 0.0])          # heading clockwise from north
        for _ in range(n_side):
            pts.append(pts[-1] + d * step)
    truth = np.array(pts)
    Ts_true, Ts_dr = [], []
    for i, p in enumerate(truth):
        h = min(i // n_side, 3) * np.pi / 2
        Ts_true.append(cs.pose_enu(p + [0, 0, -100.0], 0, 0, h))
    dr = np.zeros_like(truth)
    for i in range(1, len(truth)):                         # heading error grows with time (gyro drift): the square opens
        a = np.radians(0.12) * i
        Rb = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
        dr[i] = dr[i - 1] + Rb @ ((truth[i] - truth[i - 1]) * (1 + drift))
    for i, p in enumerate(dr):
        h = min(i // n_side, 3) * np.pi / 2
        Ts_dr.append(cs.pose_enu(p + [0, 0, -100.0], 0, 0, h))
    return np.array(Ts_true), np.array(Ts_dr)


def _graph(Ts_true, Ts_dr, with_loop=True):
    n = len(Ts_dr)
    g = pg.PoseGraph(T0=Ts_dr.copy(), t=np.arange(n, dtype=float))
    for i in range(n - 1):
        Z = cs.invert(Ts_dr[i]) @ Ts_dr[i + 1]
        g.add(pg.Edge(i, i + 1, Z, 0.3, np.radians(0.1), "odometry"))
    if with_loop:                                          # last pose coincides with the first in truth
        Z = cs.invert(Ts_true[0]) @ Ts_true[-1]
        g.add(pg.Edge(0, n - 1, Z, 0.3, np.radians(0.1), "loop_closure"))
    g.depth_meas = np.full(n, 100.0)
    g.gravity_meas = np.array([pg.gravity_vector(T[:3, :3])[:2] for T in Ts_true])
    return g


def test_pose_graph_closes_the_loop():
    Tt, Td = _square_loop()
    end_err_dr = np.linalg.norm(Td[-1][:3, 3] - Tt[-1][:3, 3])
    g = _graph(Tt, Td)
    res = opt.optimize(g, max_nfev=80)
    end_err = np.linalg.norm(res["T"][-1][:3, 3] - Tt[-1][:3, 3])
    assert end_err_dr > 5.0
    assert end_err < 0.35 * end_err_dr
    assert res["cost1"] < res["cost0"]


def test_loop_edge_is_satisfied_and_odometry_only_graph_is_unchanged():
    """A single loop closure makes the closure pair consistent (it does not by itself guarantee a lower RMS against
    truth: the whole drifted shape is only constrained at one pair). Without a loop nothing can be corrected."""
    Tt, Td = _square_loop()
    rel_err = lambda T: np.linalg.norm((cs.invert(T[0]) @ T[-1])[:3, 3] - (cs.invert(Tt[0]) @ Tt[-1])[:3, 3])
    with_loop = opt.optimize(_graph(Tt, Td, True), max_nfev=80)["T"]
    no_loop = opt.optimize(_graph(Tt, Td, False), max_nfev=80)["T"]
    assert rel_err(Td) > 5.0
    assert rel_err(with_loop) < 0.5
    assert np.allclose(no_loop[:, :3, 3], Td[:, :3, 3], atol=1e-6)


def test_many_loops_reduce_ate():
    """With several revisits (lawn-mower like) distributed corrections do reduce the trajectory error."""
    Tt, Td = _square_loop(n_side=10)
    rms = lambda X: np.sqrt(np.mean(np.sum((X[:, :3, 3] - Tt[:, :3, 3]) ** 2, 1)))
    g = _graph(Tt, Td, False)
    n = len(Td)
    for i in range(0, n, 4):                                  # absolute-like constraints via loops to the anchor
        g.add(pg.Edge(0, i, cs.invert(Tt[0]) @ Tt[i], 0.5, np.radians(0.3), "loop_closure"))
    r = opt.optimize(g, max_nfev=200)["T"]
    assert rms(r) < 0.4 * rms(Td)


def test_optimisation_is_consistent_residual_decreases():
    Tt, Td = _square_loop(drift=0.03)
    g = _graph(Tt, Td)
    ea = pg.pack_edges(g)
    r0 = pg.residuals(np.zeros(6 * g.n), g, ea)
    res = opt.optimize(g, max_nfev=60)
    r1 = pg.residuals(res["delta"], g, ea)
    assert r1 @ r1 < r0 @ r0


def test_gravity_vector_level_and_roll():
    assert np.allclose(pg.gravity_vector(cs.R_enu_body(0, 0, 0.7)), [0, 0, 1])          # down is +z in FRD
    g = pg.gravity_vector(cs.R_enu_body(np.radians(10), 0, 0))
    assert g[1] == pytest.approx(np.sin(np.radians(10)))


def test_map_reintegration_applies_corrections_rigidly():
    K = 5
    T_dr = np.tile(np.eye(4), (K, 1, 1))
    for k in range(K):
        T_dr[k, :3, 3] = [10.0 * k, 0, -100]
    T_opt = T_dr.copy()
    T_opt[:, 0, 3] += np.linspace(0, 4.0, K)               # correction grows along the path
    C = mr.keyframe_corrections(T_dr, T_opt)
    s = 10.0 * np.arange(K)
    ping_s = np.array([0.0, 5.0, 20.0, 40.0])
    Cp = mr.ping_corrections(s, C, ping_s)
    assert np.allclose(Cp[:, 0, 3], np.interp(ping_s, s, np.linspace(0, 4.0, K)), atol=1e-9)
    pts = np.array([[1.0, 2.0, -50.0]] * 4)
    new = mr.apply_to_points(pts, np.arange(4), Cp)
    assert np.allclose(new[:, 1:], pts[:, 1:]) and np.allclose(new[:, 0] - pts[:, 0], Cp[:, 0, 3])
