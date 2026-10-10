"""Sonar odometry: relative-pose constraints between consecutive keyframe submaps (ICP), accepted only when they are
consistent with the navigation prior. These are measurements; dead-reckoning odometry edges are built separately."""
import numpy as np
from scipy.spatial import cKDTree

from preprocessing.coordinate_systems import invert
from mapping.surface_normals import point_normals_pca
from .scan_matching import icp
from .pose_graph import Edge


def dr_odometry_edges(kfs, cfg_slam):
    """Relative pose from dead reckoning between consecutive keyframes with distance-scaled uncertainty."""
    edges = []
    st = cfg_slam["odom_sigma_trans_per_sqrt_m"]
    sr = np.radians(cfg_slam["odom_sigma_rot_deg_per_sqrt_m"])
    for k in range(len(kfs) - 1):
        d = max(kfs.s[k + 1] - kfs.s[k], 1.0)
        Z = invert(kfs.T0[k]) @ kfs.T0[k + 1]
        edges.append(Edge(k, k + 1, Z, st * np.sqrt(d), sr * np.sqrt(d), "odometry", float(kfs.t[k + 1])))
    return edges


def sonar_odometry_edges(kfs, cfg_slam, max_corr_m=3.0, max_corr_deg=1.0):
    edges, rejected = [], 0
    trees, normals = {}, {}
    for k in range(len(kfs) - 1):
        a, b = kfs.world_cloud(k), kfs.world_cloud(k + 1)
        if k not in trees:
            trees[k] = cKDTree(a)
            normals[k] = point_normals_pca(a, tree=trees[k])[0]
        res = icp(b, a, normals[k], max_iter=25, trim=cfg_slam["icp_trim"], max_pair_dist=cfg_slam["icp_max_pair_dist_m"],
                  tree=trees[k])
        Tc = res["T"]
        ang = np.degrees(np.arccos(np.clip((np.trace(Tc[:3, :3]) - 1) / 2, -1, 1)))
        ok = (res["inlier_fraction"] >= 0.5 and res["rmse"] <= 1.0 and np.linalg.norm(Tc[:3, 3]) <= max_corr_m
              and ang <= max_corr_deg)
        if not ok:
            rejected += 1
            continue
        Z = invert(kfs.T0[k]) @ Tc @ kfs.T0[k + 1]
        edges.append(Edge(k, k + 1, Z, 0.35, np.radians(0.15), "sonar_odometry", float(kfs.t[k + 1]),
                          {"rmse": res["rmse"], "inlier": res["inlier_fraction"]}))
    return edges, rejected
