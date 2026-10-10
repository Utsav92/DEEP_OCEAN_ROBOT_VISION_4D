"""Export everything TouchDesigner needs as memory-mappable .npy arrays + JSON (file-based playback).

TouchDesigner axes: x = East, y = Up - Z_REF, z = -North. Z_REF (metres) is subtracted from elevation so the scene sits
near y = 0 (float precision, camera clipping); it is recorded in td_meta.json and passed to shaders as a uniform.
"""
import json
import os

import numpy as np

from preprocessing.coordinate_systems import enu_to_td, T_td_enu


def to_td(p, z_ref):
    q = enu_to_td(np.asarray(p, dtype=np.float64)).copy()
    q[..., 1] -= z_ref
    return q


def pose_to_td(T, z_ref):
    """T_enu_vehicle (...,4,4) -> T_td_vehicle with the vertical reference removed."""
    out = np.einsum("ij,...jk->...ik", T_td_enu(), T)
    out[..., 1, 3] -= z_ref
    return out


def path_texture(positions_td, t, side):
    """Pack a polyline (n,3)+t into a (side,side,4) float32 texture [x,y,z,t]; padding t = 1e9 (never visible)."""
    n = min(len(positions_td), side * side)
    tex = np.zeros((side * side, 4), np.float32)
    tex[:, 3] = 1e9
    tex[:n, :3] = positions_td[:n]
    tex[:n, 3] = t[:n]
    return tex.reshape(side, side, 4)


def densify(p, t, k):
    """Linear subdivision: k points per segment (keeps time monotone)."""
    f = np.linspace(0, 1, k, endpoint=False)
    idx = np.arange(len(p) - 1)
    P = (p[idx, None, :] * (1 - f[None, :, None]) + p[idx + 1, None, :] * f[None, :, None]).reshape(-1, 3)
    T = (t[idx, None] * (1 - f[None, :]) + t[idx + 1, None] * f[None, :]).reshape(-1)
    return np.vstack([P, p[-1:]]), np.concatenate([T, t[-1:]])


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def export_all(out_dir, d):
    """d: dict with the arrays prepared by scripts/build_map.py (all in ENU)."""
    os.makedirs(out_dir, exist_ok=True)
    zr = float(d["z_ref"])
    t0 = float(d["ping_t"][0])
    meta = {"z_ref_m": zr, "t0_s": t0, "t1_s": float(d["ping_t"][-1]), "duration_s": float(d["ping_t"][-1] - t0),
            "n_pings": int(len(d["ping_t"])), "axes": "TD x=E, y=U-z_ref, z=-N", "simulated": True,
            "label": d.get("label", ""), "dataset": d.get("dataset", ""), "slam_available": bool(d["slam_available"]),
            "planned_cells": int(d.get("planned_cells", 1)), "slam": d.get("slam_stats", {})}
    # ---- sounding points (display subset, identical selection for the DR and SLAM versions)
    idx = d["display_idx"]
    N = 512 * 512
    pts_dr = np.zeros((N, 4), np.float32)
    pts_sl = np.zeros((N, 4), np.float32)
    attr = np.zeros((N, 4), np.float32)
    n = len(idx)
    pts_dr[:n, :3] = to_td(d["xyz_dr"][idx], zr)
    pts_sl[:n, :3] = to_td(d["xyz_slam"][idx], zr)
    pts_dr[:n, 3] = pts_sl[:n, 3] = np.maximum(d["rng"][idx], 0.01)    # slant range m; 0 = unused slot
    attr[:n, 0] = d["intensity"][idx]
    attr[:n, 1] = d["t"][idx] - t0                      # seconds since the first ping
    attr[:n, 2] = d["sigma_h"][idx]
    attr[:n, 3] = d["ping"][idx]
    attr[n:, 1] = 1e9
    np.save(os.path.join(out_dir, "map_points_dr.npy"), pts_dr.reshape(512, 512, 4))
    np.save(os.path.join(out_dir, "map_points_slam.npy"), pts_sl.reshape(512, 512, 4))
    np.save(os.path.join(out_dir, "map_attr.npy"), attr.reshape(512, 512, 4))
    meta["display_points"] = int(n)
    # ---- trajectories as time-tagged point paths
    side = 160
    tt = d["ping_t"] - t0
    for name, pos in (("dr", d["ping_pos_dr"]), ("slam", d["ping_pos_slam"]), ("truth", d["ping_pos_truth"])):
        P, T = densify(to_td(pos, zr), tt, 3)
        np.save(os.path.join(out_dir, f"traj_{name}.npy"), path_texture(P, T, side))
    np.save(os.path.join(out_dir, "ping_t.npy"), tt.astype(np.float64))
    np.save(os.path.join(out_dir, "ping_sigma_dr.npy"), np.asarray(d["ping_sigma_dr"], dtype=np.float32))
    np.save(os.path.join(out_dir, "vehicle_T_slam.npy"), pose_to_td(d["T_vehicle_slam"], zr).astype(np.float64))
    np.save(os.path.join(out_dir, "vehicle_T_dr.npy"), pose_to_td(d["T_vehicle_dr"], zr).astype(np.float64))
    # ---- beams per ping (sonar origin + valid returns)
    np.save(os.path.join(out_dir, "beam_end_slam.npy"), d["beam_end_td"].astype(np.float32))
    np.save(os.path.join(out_dir, "beam_origin_slam.npy"), d["beam_origin_td"].astype(np.float32))
    np.save(os.path.join(out_dir, "beam_theta.npy"), d["beam_theta"].astype(np.float32))
    # ---- reconstruction texture [z_rel, t_first_s, sigma, count], plus survey extent
    np.save(os.path.join(out_dir, "recon_tex.npy"), d["recon_tex"].astype(np.float32))
    meta["recon_extent_xxyy"] = [float(v) for v in d["recon_extent"]]
    meta["recon_shape"] = list(d["recon_tex"].shape)
    # ---- loop closures / versions / objects
    loops = [{"i": int(i), "j": int(j), "t_s": float(t - t0), "p_i": to_td(pi, zr).tolist(), "p_j": to_td(pj, zr).tolist(),
              "rmse_m": float(r)} for i, j, t, pi, pj, r in d["loops"]]
    write_json(os.path.join(out_dir, "loops.json"), loops)
    V = d["versions"]
    np.savez_compressed(os.path.join(out_dir, "versions.npz"), K=V["K"], t_end=V["t_end"] - t0, n_loops=V["n_loops"],
                        traj=to_td(V["pos"], zr).astype(np.float32), kf_t=d["kf_t"] - t0)
    write_json(os.path.join(out_dir, "objects.json"), {"candidates": d["candidates"], "inserted_synthetic": d["truth_objects"]})
    meta["n_loops"] = len(loops)
    meta["n_versions"] = int(len(V["K"]))
    write_json(os.path.join(out_dir, "td_meta.json"), meta)
    return meta
