"""Scan matching for 3D sonar point clouds: masked normalised cross-correlation of bathymetric rasters (coarse,
TERCOM-like) and trimmed point-to-plane ICP (fine) with an information-matrix output for observability checks."""
import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mapping.surface_normals import point_normals_pca


def small_twist_to_T(w, v):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(w).as_matrix()
    T[:3, 3] = v
    return T


def rasterize(points, x0, y0, cell, shape):
    """Mean z per cell; returns (z (ny,nx) with 0 where empty, mask (ny,nx) float)."""
    ix = np.floor((points[:, 0] - x0) / cell).astype(int)
    iy = np.floor((points[:, 1] - y0) / cell).astype(int)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    s = np.zeros(shape)
    n = np.zeros(shape)
    np.add.at(s, (iy[ok], ix[ok]), points[ok, 2])
    np.add.at(n, (iy[ok], ix[ok]), 1.0)
    m = n > 0
    z = np.where(m, s / np.maximum(n, 1), 0.0)
    return z, m.astype(np.float64)


def masked_ncc(A, MA, B, MB, max_shift_cells, min_overlap_cells=60):
    """Normalised cross-correlation of B shifted over A (both with validity masks), shifts within +-max_shift_cells.
    Returns (ncc [2m+1, 2m+1] with NaN where overlap too small, shifts axis in cells). ncc[dy+m, dx+m] corresponds to
    moving B by (+dx, +dy) cells (x = columns, y = rows)."""
    m = int(max_shift_cells)
    ones = np.ones((1, 1))
    Bf, MBf = B[::-1, ::-1], MB[::-1, ::-1]
    BBf = (B * B * MB)[::-1, ::-1]

    def corr(P, Qf):                              # sum over overlap of P(x) Q(x - s)
        return fftconvolve(P, Qf, mode="full")

    N = corr(MA, MBf)
    SA = corr(A * MA, MBf)
    SB = corr(MA, (B * MB)[::-1, ::-1])
    SAB = corr(A * MA, (B * MB)[::-1, ::-1])
    SAA = corr(A * A * MA, MBf)
    SBB = corr(MA, BBf)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = SAB - SA * SB / N
        va = SAA - SA * SA / N
        vb = SBB - SB * SB / N
        ncc = cov / np.sqrt(np.maximum(va * vb, 1e-12))
    ncc = np.where(N >= min_overlap_cells, ncc, np.nan)
    hb, wb = B.shape
    # full-mode index (r, c) <-> B shifted by (r-(hb-1), c-(wb-1))
    cy, cx = hb - 1, wb - 1
    sub = ncc[cy - m: cy + m + 1, cx - m: cx + m + 1]
    # full correlation with flipped B: output at (cy+dy, cx+dx) = overlap when B is moved by (dx, dy) in the positive direction
    return sub


def best_shift(ncc, m, exclude_cells=2):
    """Peak shift (dx, dy in cells), peak value, and the best value outside an exclusion neighbourhood of the peak
    (ratio test for ambiguity)."""
    v = np.where(np.isfinite(ncc), ncc, -2.0)
    iy, ix = np.unravel_index(np.argmax(v), v.shape)
    peak = float(v[iy, ix])
    yy, xx = np.ogrid[:v.shape[0], :v.shape[1]]
    far = (np.abs(yy - iy) > exclude_cells) | (np.abs(xx - ix) > exclude_cells)
    second = float(np.max(np.where(far, v, -2.0)))
    return (ix - m, iy - m), peak, second


def icp(src, dst, dst_normals=None, init=None, max_iter=40, trim=0.8, max_pair_dist=12.0, tree=None, tol=1e-4):
    """Point-to-plane ICP aligning `src` onto `dst` (both (n,3) world-frame). Returns dict:
    T (4x4 applied to src), rmse, inlier_fraction, n_pairs, info (6x6 [rot,trans] information), converged."""
    T = np.eye(4) if init is None else init.copy()
    if tree is None:
        tree = cKDTree(dst)
    if dst_normals is None:
        dst_normals, _, _ = point_normals_pca(dst, tree=tree)
    prev = np.inf
    info = np.zeros((6, 6))
    rmse, inl, overlap, npairs, conv = np.inf, 0.0, 0.0, 0, False
    for it in range(max_iter):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(p, distance_upper_bound=max_pair_dist)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        idx = np.where(ok)[0]
        dd = d[idx]
        keep = idx[dd <= np.quantile(dd, trim)]
        q, n, pk = dst[j[keep]], dst_normals[j[keep]], p[keep]
        r = np.einsum("ij,ij->i", n, pk - q)
        J = np.hstack([np.cross(pk, n), n])               # twist [omega, v] applied in the world frame
        H = J.T @ J + 1e-6 * np.eye(6)
        g = J.T @ r
        delta = -np.linalg.solve(H, g)
        # statistics at the pose BEFORE this update, over all matched pairs (not only the trimmed set)
        r_all = np.einsum("ij,ij->i", dst_normals[j[idx]], p[idx] - dst[j[idx]])
        overlap = float(len(idx) / len(src))
        inl = float(np.mean(np.abs(r_all) < max(1.0, 2.5 * np.sqrt(np.mean(r ** 2)))))
        T = small_twist_to_T(delta[:3], delta[3:]) @ T
        rmse = float(np.sqrt(np.mean(r ** 2)))
        npairs = int(len(keep))
        info = J.T @ J
        if abs(prev - rmse) < tol and np.linalg.norm(delta) < 1e-3:
            conv = True
            break
        prev = rmse
    return {"T": T, "rmse": rmse, "inlier_fraction": inl, "overlap": overlap, "n_pairs": npairs, "info": info,
            "converged": conv}


def info_min_eig_translation(info):
    return float(np.linalg.eigvalsh(info[3:, 3:])[0])


def predicted_translation_sigma(info, rmse, floor=0.3):
    """Cramer-Rao style 1-sigma of the weakest translation direction: sigma_r / sqrt(lambda_min)."""
    lam = max(info_min_eig_translation(info), 1e-9)
    return float(max(rmse, floor) / np.sqrt(lam))


def info_condition(info):
    w = np.linalg.eigvalsh(info)
    return float(w[-1] / max(w[0], 1e-12))
