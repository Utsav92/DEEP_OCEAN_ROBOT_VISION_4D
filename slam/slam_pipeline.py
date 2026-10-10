"""End-to-end underwater SLAM on the dead-reckoned soundings:
keyframes -> DR odometry + sonar odometry + verified loop closures -> robust pose graph (incremental versions)
-> map reintegration. Falls back to navigation-only mapping, explicitly marked, when overlap is insufficient."""
import json
import os
import pickle
import time

import numpy as np

from preprocessing.config import abspath, ensure_dir
from preprocessing.coordinate_systems import R_enu_body
from . import sonar_features, sonar_odometry, loop_closure, optimization, map_reintegration as mr
from .pose_graph import PoseGraph, gravity_vector


def _build_graph(kfs, nav, base_edges, cfg_slam):
    T0 = kfs.T0
    kt = kfs.t
    r = np.interp(kt, nav["t"], nav["roll"])
    p = np.interp(kt, nav["t"], nav["pitch"])
    y = np.unwrap(nav["yaw"])
    y = np.interp(kt, nav["t"], y)
    depth = np.interp(kt, nav["t"], nav["depth"])
    grav = np.array([gravity_vector(R_enu_body(a, b, c))[:2] for a, b, c in zip(r, p, y)])
    g = PoseGraph(T0=T0.copy(), t=kt.copy(), edges=list(base_edges), depth_meas=depth, gravity_meas=grav,
                  yaw_meas=y, sigma_depth=cfg_slam["prior_sigma_z_m"], sigma_tilt=np.radians(cfg_slam["prior_sigma_tilt_deg"]))
    return g


def _subgraph(g, K, edges):
    sg = PoseGraph(T0=g.T0[:K], t=g.t[:K], edges=[e for e in edges if max(e.i, e.j) < K],
                   depth_meas=g.depth_meas[:K], gravity_meas=g.gravity_meas[:K], yaw_meas=g.yaw_meas[:K],
                   sigma_depth=g.sigma_depth, sigma_tilt=g.sigma_tilt, sigma_yaw=g.sigma_yaw)
    return sg


def _edge_chi(g, T, edges):
    from .pose_graph import inv_batch
    from scipy.spatial.transform import Rotation
    i = np.array([e.i for e in edges]); j = np.array([e.j for e in edges])
    Z = np.stack([e.Z for e in edges])
    E = inv_batch(Z) @ inv_batch(T[i]) @ T[j]
    rt = E[:, :3, 3] / np.array([e.sigma_t for e in edges])[:, None]
    rr = Rotation.from_matrix(E[:, :3, :3]).as_rotvec() / np.array([e.sigma_r for e in edges])[:, None]
    return np.sqrt((rt ** 2).sum(1) + (rr ** 2).sum(1))


def run(cfg, log=print, reuse_loops=False):
    cs = cfg["slam"]
    proc = abspath(cfg, cfg["paths"]["processed"])
    sim = abspath(cfg, cfg["paths"]["simulation"])
    diag = ensure_dir(os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics"))
    snd = dict(np.load(os.path.join(proc, "soundings_dr.npz")))
    nav_dr = np.load(os.path.join(proc, "nav_dr.npz"))
    nav = dict(np.load(os.path.join(sim, "nav_measured.npz")))
    ping_t = snd["ping_t"]
    ping_s = np.interp(ping_t, nav_dr["t"], nav_dr["distance"])
    ping_sig = np.interp(ping_t, nav_dr["t"], nav_dr["sigma_xy"])
    t0 = time.time()
    kfs = sonar_features.build_keyframes(snd, ping_s, ping_sig, cs["keyframe_distance_m"], 60.0)
    log(f"keyframes: {len(kfs)}  (every {cs['keyframe_distance_m']} m, submap +-60 m)")
    odo = sonar_odometry.dr_odometry_edges(kfs, cs)
    sod, rej = sonar_odometry.sonar_odometry_edges(kfs, cs)
    log(f"odometry edges {len(odo)}, sonar-odometry edges {len(sod)} (rejected {rej})  [{time.time() - t0:.0f}s]")
    cache = os.path.join(proc, "loops_cache.pkl")
    if reuse_loops and os.path.exists(cache):
        loops, reasons, npairs = pickle.load(open(cache, "rb"))
        log(f"loop closures reloaded from cache ({len(loops)})")
    else:
        loops, reasons, npairs = loop_closure.detect(
            kfs, cs, progress=lambda n, m, a: log(f"  loop candidates {n}/{m}, accepted {a}") if n else None)
        pickle.dump((loops, reasons, npairs), open(cache, "wb"))
    log(f"loop closures accepted {len(loops)} of {npairs} candidates; rejected: "
        f"{ {k: v for k, v in reasons.items() if k != 'accepted'} }  [{time.time() - t0:.0f}s]")
    report = {"n_keyframes": len(kfs), "odometry_edges": len(odo), "sonar_odometry_edges": len(sod),
              "sonar_odometry_rejected": rej, "loop_candidates": npairs, "loop_accepted": len(loops),
              "loop_rejection_reasons": reasons, "simulated": True}
    slam_available = len(loops) >= 5
    report["slam_available"] = slam_available
    g = _build_graph(kfs, nav, odo + sod, cs)
    # Diagnostic only: loops that stay inconsistent after a global solve. They are NOT dropped - a correct loop that
    # needs a large correction is inconsistent with the rigid odometry chain until the optimiser has moved it, so
    # residual-based screening removes exactly the most valuable closures (observed: ATE gain disappeared).
    n_screened = 0
    if slam_available:
        res0 = optimization.optimize(g, edges=odo + sod + loops, max_nfev=cs["optimizer_max_nfev"])
        chi = _edge_chi(g, res0["T"], loops)
        n_screened = int((chi > 4.0).sum())
        log(f"  {n_screened} of {len(loops)} loop closures remain inconsistent (chi > 4) after a global solve (kept; robust loss)")
    report["loops_screened_out"] = n_screened
    report["loop_accepted_after_screening"] = len(loops)
    all_edges = odo + sod + (loops if slam_available else [])
    # incremental map versions
    K = len(kfs)
    cps = sorted(set(np.linspace(K // 10, K, 10).astype(int)))
    versions, x_prev = [], None
    for k in cps:
        sg = _subgraph(g, k, all_edges)
        x0 = None if x_prev is None else np.concatenate([x_prev, np.zeros(6 * (k - len(x_prev) // 6))])
        res = optimization.optimize(sg, x0=x0, max_nfev=cs["optimizer_max_nfev"])
        x_prev = res["delta"]
        versions.append({"K": int(k), "T": res["T"].astype(np.float64), "t_end": float(kfs.t[k - 1]),
                         "n_loops": int(sum(1 for e in sg.edges if e.kind == "loop_closure")),
                         "cost0": res["cost0"], "cost1": res["cost1"], "seconds": res["seconds"]})
        log(f"  version {len(versions)}: K={k} loops={versions[-1]['n_loops']} cost {res['cost0']:.0f}->{res['cost1']:.0f} "
            f"({res['seconds']:.1f}s)")
    T_opt = versions[-1]["T"]
    # reintegration
    C = mr.keyframe_corrections(kfs.T0, T_opt)
    Cp = mr.ping_corrections(kfs.s, C, ping_s)
    xyz_new = mr.apply_to_points(snd["xyz"].astype(np.float64), snd["ping"], Cp)
    Tv_new = mr.correct_poses(snd["T_world_vehicle"], Cp)
    Ts_new = mr.correct_poses(snd["T_world_sonar"], Cp)
    out = os.path.join(proc, "slam_result.npz")
    np.savez_compressed(
        out, kf_ping=kfs.ping, kf_t=kfs.t, kf_s=kfs.s, kf_T_dr=kfs.T0, kf_T_opt=T_opt,
        xyz_slam=xyz_new.astype(np.float32), T_world_vehicle_slam=Tv_new, T_world_sonar_slam=Ts_new,
        loop_i=np.array([e.i for e in loops], int), loop_j=np.array([e.j for e in loops], int),
        loop_t=np.array([e.t for e in loops]), loop_rmse=np.array([e.info["rmse"] for e in loops]),
        loop_shift=np.array([e.info["shift_m"] for e in loops]).reshape(-1, 2),
        version_K=np.array([v["K"] for v in versions]), version_t=np.array([v["t_end"] for v in versions]),
        version_loops=np.array([v["n_loops"] for v in versions]),
        version_T=np.concatenate([np.pad(v["T"], ((0, K - v["K"]), (0, 0), (0, 0)), mode="edge")[None] for v in versions]),
        slam_available=slam_available)
    report["seconds"] = time.time() - t0
    report["versions"] = [{k: v[k] for k in ("K", "t_end", "n_loops", "cost0", "cost1", "seconds")} for v in versions]
    with open(os.path.join(diag, "slam_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    return report
