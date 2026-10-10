"""Loop closure: candidate generation, coarse correlation, fine ICP and strict geometric verification.

A candidate becomes an accepted loop closure only if ALL of these hold:
  * coarse masked-NCC peak >= ncc_min and not ambiguous (peak / runner-up > ratio)
  * ICP inlier fraction >= loop_min_inlier_fraction and RMSE <= loop_max_rmse_m
  * translation information (smallest eigenvalue) >= loop_min_eigenvalue  (observable, not a featureless corridor)
  * several perturbed starts converge to the same solution (spread <= loop_max_multistart_spread_m)
  * the reverse registration agrees (symmetry) within loop_symmetry_tol_m
  * the correction is within a gate derived from the navigation uncertainty
Everything rejected is counted with its reason; nothing else is ever displayed as a loop closure.
"""
import numpy as np
from scipy.spatial import cKDTree

from preprocessing.coordinate_systems import invert
from mapping.surface_normals import point_normals_pca
from .pose_graph import Edge
from .scan_matching import rasterize, masked_ncc, best_shift, icp, info_min_eig_translation, predicted_translation_sigma


def footprints(kfs, cell=12.0):
    """Coverage footprint of every keyframe submap as a sorted array of integer cell keys (dead-reckoned world frame)."""
    out = []
    for k in range(len(kfs)):
        w = kfs.world_cloud(k)
        ix = np.floor(w[:, 0] / cell).astype(np.int64) + 100000
        iy = np.floor(w[:, 1] / cell).astype(np.int64) + 100000
        out.append(np.unique(ix * 400000 + iy))
    return out


def candidates(kfs, cfg, max_per_kf=None, stride=2, min_footprint_overlap=0.25):
    """(i, j) pairs, i<j, far apart along the path but with overlapping acoustic footprints (best overlaps first,
    spread along the other line)."""
    pos = kfs.T0[:, :3, 3]
    tree = cKDTree(pos[:, :2])
    R = cfg["loop_search_radius_m"] + 160.0
    gap = cfg["loop_min_index_gap"]
    mp = max_per_kf or cfg["loop_max_candidates_per_kf"]
    fp = footprints(kfs)
    pairs = []
    for j in range(0, len(kfs), stride):
        near = [i for i in tree.query_ball_point(pos[j, :2], R + 2 * kfs.sigma_xy[j] * 0.3) if i < j - gap]
        scored = []
        for i in near:
            ov = len(np.intersect1d(fp[i], fp[j], assume_unique=True)) / max(min(len(fp[i]), len(fp[j])), 1)
            if ov >= min_footprint_overlap:
                scored.append((ov, i))
        scored.sort(reverse=True)
        chosen = []
        for ov, i in scored:
            if all(abs(i - c) >= 4 for c in chosen):
                chosen.append(i)
            if len(chosen) >= mp:
                break
        pairs += [(i, j) for i in chosen]
    return pairs


def _ncc_stage(ci, cj, cell, max_shift_m):
    allp = np.vstack([ci, cj])
    x0 = np.floor(allp[:, 0].min() / cell) * cell - max_shift_m
    y0 = np.floor(allp[:, 1].min() / cell) * cell - max_shift_m
    nx = int(np.ceil((allp[:, 0].max() + max_shift_m - x0) / cell)) + 1
    ny = int(np.ceil((allp[:, 1].max() + max_shift_m - y0) / cell)) + 1
    A, MA = rasterize(ci, x0, y0, cell, (ny, nx))
    B, MB = rasterize(cj, x0, y0, cell, (ny, nx))
    m = int(round(max_shift_m / cell))
    ncc = masked_ncc(A, MA, B, MB, m, min_overlap_cells=40)
    return best_shift(ncc, m)


def try_pair(kfs, i, j, cfg, trees=None):
    """Returns (Edge or None, reason string)."""
    ci_w, cj_w = kfs.world_cloud(i), kfs.world_cloud(j)
    c0 = ci_w.mean(0)                                  # registration is done about the patch centre (no long lever arm)
    ci, cj = ci_w - c0, cj_w - c0
    cell = cfg["grid_cell_m"]
    max_shift = min(60.0, 40.0 + 3 * max(kfs.sigma_xy[i], kfs.sigma_xy[j]) * 0.25 + 10.0)
    (dx, dy), peak, second = _ncc_stage(ci, cj, cell, max_shift)
    if peak < 0.55:
        return None, "ncc_low"
    if second > 0.97 * peak - 0.0 and peak < 0.9:
        return None, "ncc_ambiguous"
    if trees is not None and i in trees:
        tree, nrm = trees[i]
    else:
        tree = cKDTree(ci)
        nrm = point_normals_pca(ci, tree=tree)[0]
        if trees is not None:
            trees[i] = (tree, nrm)
    init = np.eye(4)
    init[0, 3], init[1, 3] = dx * cell, dy * cell
    res = icp(cj, ci, nrm, init=init, max_iter=cfg["icp_max_iter"], trim=cfg["icp_trim"],
              max_pair_dist=cfg["icp_max_pair_dist_m"], tree=tree)
    if res["overlap"] < cfg["loop_min_overlap"]:
        return None, "icp_overlap"
    if res["inlier_fraction"] < cfg["loop_min_inlier_fraction"]:
        return None, "icp_inliers"
    if res["rmse"] > cfg["loop_max_rmse_m"]:
        return None, "icp_rmse"
    if predicted_translation_sigma(res["info"], res["rmse"]) > cfg["loop_max_pred_sigma_m"]:
        return None, "unobservable"
    # multi-start basin test
    sols = [res["T"][:3, 3]]
    for off in ((6.0, 0.0), (-6.0, 0.0), (0.0, 6.0), (0.0, -6.0)):
        s = np.eye(4)
        s[0, 3], s[1, 3] = init[0, 3] + off[0], init[1, 3] + off[1]
        r2 = icp(cj, ci, nrm, init=s, max_iter=cfg["icp_max_iter"], trim=cfg["icp_trim"],
                 max_pair_dist=cfg["icp_max_pair_dist_m"], tree=tree)
        if r2["rmse"] <= cfg["loop_max_rmse_m"] * 1.5 and r2["inlier_fraction"] >= 0.7 * cfg["loop_min_inlier_fraction"]:
            sols.append(r2["T"][:3, 3])
    sols = np.array(sols)
    if len(sols) < 3 or np.max(np.linalg.norm(sols - sols.mean(0), axis=1)) > cfg["loop_max_multistart_spread_m"]:
        return None, "multistart_ambiguous"
    # symmetry: register i onto j and compare
    tj = cKDTree(cj)
    nj = point_normals_pca(cj, tree=tj)[0]
    back = icp(ci, cj, nj, init=np.linalg.inv(res["T"]), max_iter=cfg["icp_max_iter"], trim=cfg["icp_trim"],
               max_pair_dist=cfg["icp_max_pair_dist_m"], tree=tj)
    if np.linalg.norm((back["T"] @ res["T"])[:3, 3]) > cfg["loop_symmetry_tol_m"]:
        return None, "asymmetric"
    corr = res["T"].copy()
    corr[:3, 3] = corr[:3, 3] + c0 - corr[:3, :3] @ c0       # back to a world-frame transform p' = R p + t
    ang = np.degrees(np.arccos(np.clip((np.trace(corr[:3, :3]) - 1) / 2, -1, 1)))
    gate = 15.0 + 3.0 * max(kfs.sigma_xy[i], kfs.sigma_xy[j]) * 0.5
    if np.linalg.norm(corr[:2, 3]) > max(gate, 80.0) or abs(corr[2, 3]) > 6.0 or ang > 6.0:
        return None, "gate"
    Z = invert(kfs.T0[i]) @ corr @ kfs.T0[j]
    e = Edge(int(i), int(j), Z, max(1.0, 2.0 * res["rmse"]), np.radians(0.5), "loop_closure", float(kfs.t[j]),
             {"rmse": res["rmse"], "inlier": res["inlier_fraction"], "ncc": peak, "ncc_second": second,
              "min_eig": info_min_eig_translation(res["info"]), "overlap": res["overlap"], "shift_m": [float(corr[0, 3] + corr[:3, :3][0] @ c0 - c0[0]), float(corr[1, 3] + corr[:3, :3][1] @ c0 - c0[1])]})
    return e, "accepted"


def detect(kfs, cfg, progress=None):
    pairs = candidates(kfs, cfg)
    trees = {}
    edges, reasons = [], {}
    for n, (i, j) in enumerate(pairs):
        e, why = try_pair(kfs, i, j, cfg, trees)
        reasons[why] = reasons.get(why, 0) + 1
        if e is not None:
            edges.append(e)
        if progress and n % 200 == 0:
            progress(n, len(pairs), len(edges))
        if len(trees) > 300:
            trees.clear()
    return edges, reasons, len(pairs)
