import numpy as np
import pytest
from scipy.spatial import cKDTree

from preprocessing import coordinate_systems as cs
from slam import scan_matching as sm
from mapping.surface_normals import point_normals_pca


def _terrain(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-60, 60, (n, 2))
    z = 6 * np.sin(xy[:, 0] / 17.0) + 5 * np.cos(xy[:, 1] / 13.0) + 0.02 * xy[:, 0] * xy[:, 1] / 30
    return np.column_stack([xy, z])


def test_masked_ncc_recovers_known_shift():
    rng = np.random.default_rng(0)
    A = np.cumsum(np.cumsum(rng.normal(size=(80, 80)), 0), 1)
    A = (A - A.mean()) / A.std()
    dx, dy = 5, -3
    B = np.roll(np.roll(A, dy, 0), dx, 1)                   # B content moved by (+dx,+dy) relative to A
    M = np.ones_like(A)
    ncc = sm.masked_ncc(A, M, B, M, 10, min_overlap_cells=500)
    (sx, sy), peak, second = sm.best_shift(ncc, 10)
    # moving B by (sx, sy) re-aligns it with A: expect (-dx, -dy)
    assert (sx, sy) == (-dx, -dy)
    assert peak > 0.9 and second < peak


def test_icp_recovers_rigid_transform():
    P = _terrain()
    T_true = cs.make_T(cs.rot_z(np.radians(2.5)) @ cs.rot_x(np.radians(0.5)), [3.0, -2.0, 0.6])
    Q = cs.transform_points(T_true, P)                       # moved copy
    rng = np.random.default_rng(3)
    Q += rng.normal(0, 0.05, Q.shape)
    res = sm.icp(Q, P, init=np.eye(4), max_iter=60)           # find T that maps Q back onto P
    Tinv = cs.invert(T_true)
    err_t = np.linalg.norm(res["T"][:3, 3] - Tinv[:3, 3])
    err_r = np.degrees(np.arccos(np.clip((np.trace(res["T"][:3, :3].T @ Tinv[:3, :3]) - 1) / 2, -1, 1)))
    assert err_t < 0.15 and err_r < 0.15
    assert res["rmse"] < 0.2 and res["inlier_fraction"] > 0.9 and res["overlap"] > 0.9


def test_icp_partial_overlap_statistics():
    P = _terrain()
    left = P[P[:, 0] < 15]
    right = P[P[:, 0] > -15]
    res = sm.icp(right, left, max_iter=30)
    assert 0.3 < res["overlap"] < 0.8                         # only part of `right` has a partner in `left`


def test_flat_corridor_is_flagged_unobservable():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-50, 50, (3000, 2))
    flat = np.column_stack([xy, 0.02 * rng.normal(size=3000)])
    res = sm.icp(flat + [0.5, 0.0, 0.0], flat, max_iter=15)
    assert sm.predicted_translation_sigma(res["info"], res["rmse"]) > 1.5      # horizontal slide is unconstrained


def test_rich_terrain_is_observable():
    P = _terrain()
    res = sm.icp(P + [0.3, 0.2, 0.0], P, max_iter=30)
    assert sm.predicted_translation_sigma(res["info"], res["rmse"]) < 0.5


def test_pca_normals_upward_on_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-10, 10, (500, 2))
    n, curv, _ = point_normals_pca(np.column_stack([xy, np.zeros(500)]))
    assert np.all(n[:, 2] > 0.99) and curv.max() < 1e-6
