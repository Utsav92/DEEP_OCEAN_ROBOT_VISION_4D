"""Pose-graph optimisation (robust nonlinear least squares, scipy trf with sparse finite-difference Jacobian).
GTSAM is not installed in this environment; this is a compatible Gauss-Newton-style solver on the same SE(3) factors."""
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .pose_graph import residuals, pack_edges, poses_from_delta


def _sparsity(g, ea, n_res):
    """Jacobian sparsity pattern: which node variables each residual block touches."""
    n = g.n
    rows = []
    S = lil_matrix((n_res, 6 * n), dtype=np.int8)
    r = 0
    ne = len(ea["i"])
    cols_of = lambda k: range(6 * k, 6 * k + 6)
    if ne:
        for blk in range(2):                               # translation block, rotation block
            for e in range(ne):
                for k in (ea["i"][e], ea["j"][e]):
                    for c in cols_of(k):
                        S[r + e * 3:r + e * 3 + 3, c] = 1
            r += 3 * ne
    for count in (n if g.depth_meas is not None else 0, 2 * n if g.gravity_meas is not None else 0,
                  n if g.yaw_meas is not None else 0):
        per = count // n if n and count else 0
        for k in range(n):
            for c in cols_of(k):
                S[r + k * per:r + k * per + per, c] = 1
        r += count
    for c in cols_of(g.anchor):
        S[r:r + 6, c] = 1
    return S


def optimize(g, x0=None, edges=None, max_nfev=60, robust_scale=3.0, verbose=0):
    """Returns dict(T (n,4,4) optimised poses, delta, cost0, cost1, nfev, seconds, n_edges)."""
    ea = pack_edges(g, edges)
    x0 = np.zeros(6 * g.n) if x0 is None else x0
    n_res = len(residuals(x0, g, ea))
    S = _sparsity(g, ea, n_res)
    t0 = time.time()
    r0 = residuals(x0, g, ea)
    sol = least_squares(residuals, x0, jac_sparsity=S, args=(g, ea), loss="huber", f_scale=robust_scale,
                        method="trf", x_scale="jac", max_nfev=max_nfev, verbose=verbose, xtol=1e-8, ftol=1e-8)
    return {"T": poses_from_delta(g.T0, sol.x), "delta": sol.x, "cost0": float(0.5 * r0 @ r0),
            "cost1": float(sol.cost), "nfev": int(sol.nfev), "seconds": time.time() - t0,
            "n_edges": int(len(ea["i"])), "status": int(sol.status)}
