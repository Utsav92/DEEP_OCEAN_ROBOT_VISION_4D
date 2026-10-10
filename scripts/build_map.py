"""Stage 7: persistent map from (SLAM-corrected) soundings: gridding, terrain mesh, contours, coverage, uncertainty,
occupancy, obstacle candidates, evaluation against the simulated truth, and the TouchDesigner export.

    python scripts/build_map.py --config config.yaml
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath, ensure_dir                        # noqa: E402
from mapping.bathymetry_grid import grid_soundings                                      # noqa: E402
from mapping.terrain_mesh import grid_triangles, vertex_normals, export_mesh_bundle     # noqa: E402
from mapping.depth_contours import contours, contour_segments_enu                       # noqa: E402
from mapping.pointcloud_fusion import select_display_subset                             # noqa: E402
from mapping.occupancy_map import OccupancyMap                                          # noqa: E402
from mapping import obstacle_candidates as oc, uncertainty_map as um                    # noqa: E402
from mapping.map_export import resample_to_texture                                      # noqa: E402
from preprocessing.coordinate_systems import enu_to_td                                  # noqa: E402
from sonar.acoustic_projection import beam_fan_segments                                 # noqa: E402
from temporal_4d.temporal_export import export_all, to_td                               # noqa: E402


def grid_vs_truth(grid, truth_x, truth_y, truth_z):
    """Elevation error of OBSERVED cells against the truth terrain (bilinear sample at cell centres)."""
    from scipy.ndimage import map_coordinates
    X, Y = np.meshgrid(grid.x, grid.y)
    cx = (X - truth_x[0]) / (truth_x[1] - truth_x[0])
    cy = (Y - truth_y[0]) / (truth_y[1] - truth_y[0])
    zt = map_coordinates(truth_z, [cy, cx], order=1, mode="nearest")
    m = grid.observed
    e = grid.z[m] - zt[m]
    return {"cells": int(m.sum()), "rmse_m": float(np.sqrt(np.mean(e ** 2))), "mae_m": float(np.mean(np.abs(e))),
            "bias_m": float(e.mean()), "p95_abs_m": float(np.percentile(np.abs(e), 95))}


def build(cfg):
    mc, tc = cfg["mapping"], cfg["temporal"]
    proc, sim = abspath(cfg, cfg["paths"]["processed"]), abspath(cfg, cfg["paths"]["simulation"])
    out = ensure_dir(abspath(cfg, cfg["paths"]["output"]))
    snd = np.load(os.path.join(proc, "soundings_dr.npz"))
    sl = np.load(os.path.join(proc, "slam_result.npz"))
    tr = np.load(os.path.join(sim, "truth.npz"))
    tt = np.load(os.path.join(sim, "truth_terrain.npz"))
    meta_sim = json.load(open(os.path.join(sim, "sim_meta.json")))
    slam_ok = bool(sl["slam_available"])
    xyz_dr = snd["xyz"].astype(np.float64)
    xyz_sl = sl["xyz_slam"].astype(np.float64) if slam_ok else xyz_dr
    ext = (float(tt["x"][0]), float(tt["x"][-1]), float(tt["y"][0]), float(tt["y"][-1]))
    sp = mc["grid_spacing_m"]
    args = (snd["t"], snd["intensity"], snd["sigma_v"], snd["sigma_h"], sp, ext)
    g_dr = grid_soundings(xyz_dr, *args)
    g_sl = grid_soundings(xyz_sl, *args)
    ev = {"dead_reckoning": grid_vs_truth(g_dr, tt["x"], tt["y"], tt["z"]),
          "slam": grid_vs_truth(g_sl, tt["x"], tt["y"], tt["z"]),
          "note": "evaluated on observed cells against the SIMULATED truth terrain (real bathymetry + inserted objects)"}
    # survey coverage of the planned survey rectangle
    hx, hy = meta_sim["half_extent_m"]
    X, Y = np.meshgrid(g_sl.x, g_sl.y)
    rect = (np.abs(X) <= hx) & (np.abs(Y) <= hy)
    cov = g_sl.coverage_fraction(rect)
    # mesh of OBSERVED cells only (interpolation is exported separately and flagged)
    zobs = np.where(g_sl.observed, g_sl.z, np.nan)
    valid = np.isfinite(zobs)
    V = np.stack([X, Y, np.where(valid, zobs, 0.0)], -1).reshape(-1, 3)
    T = grid_triangles(valid)
    mesh = {"vertices": V.astype(np.float32), "triangles": T, "normals": vertex_normals(V, T).astype(np.float32),
            "valid": valid, "x": g_sl.x, "y": g_sl.y}
    mmeta = export_mesh_bundle(mesh, os.path.join(out, "meshes"), "reconstructed_seabed",
                               {"frame": "local ENU m", "kind": "observed cells only (no interpolation)",
                                "source": "SIMULATED multibeam over real NOAA bathymetry", "slam": slam_ok})
    cs = contours(g_sl.x, g_sl.y, np.where(g_sl.observed, g_sl.z, np.nan), mc["contour_interval_m"])
    seg = contour_segments_enu(cs)
    np.save(os.path.join(out, "terrain", "contours_enu.npy"), seg.astype(np.float32))
    # obstacle candidates
    cands, top = oc.detect(g_sl, mc["obstacle_opening_m"], mc["obstacle_min_height_m"], mc["obstacle_min_area_m2"],
                           "multibeam (simulated)")
    obs_eval = oc.match_to_truth(cands, meta_sim["objects"])
    # occupancy from ray geometry (slam corrected sonar origins / returns)
    Ts = sl["T_world_sonar_slam"] if slam_ok else snd["T_world_sonar"]
    occ = OccupancyMap((ext[0], ext[1], ext[2], ext[3]), (float(tt["z"].min()) - 10, 0.0), mc["voxel_m"])
    sub = np.arange(0, len(snd["ping"]), 2)
    occ.integrate(Ts[snd["ping"][sub], :3, 3], xyz_sl[sub])
    occ_stats = occ.stats()
    # reconstruction texture (for the TouchDesigner terrain growth shader)
    size = 512
    zt, ok, extent = resample_to_texture(g_sl.x, g_sl.y, np.where(g_sl.observed, g_sl.z, np.nan), size, ext)
    tf, _, _ = resample_to_texture(g_sl.x, g_sl.y, np.where(g_sl.observed, g_sl.t_first, np.nan), size, ext)
    sg, _, _ = resample_to_texture(g_sl.x, g_sl.y, np.where(g_sl.observed, g_sl.sigma, np.nan), size, ext)
    cn, _, _ = resample_to_texture(g_sl.x, g_sl.y, np.where(g_sl.observed, g_sl.count.astype(float), np.nan), size, ext)
    t0 = float(snd["ping_t"][0])
    # reconstructed height texture: elevation where observed, otherwise the sentinel -1e4
    recon = np.stack([np.where(np.isfinite(zt), zt, -1e4), np.where(np.isfinite(tf), tf - t0, 1e9),
                      np.where(np.isfinite(sg), sg, 0.0), np.where(np.isfinite(cn), cn, 0.0)], -1)
    # display subset and per-ping beam geometry
    idx = select_display_subset(len(xyz_sl), mc["max_display_points"])
    z_ref = float(np.round(np.nanmedian(tt["z"])))
    P = len(snd["ping_t"])
    B = int(json.load(open(os.path.join(sim, "sim_meta.json")))["n_beams"])
    beam_end = np.zeros((P, B, 4), np.float32)
    beam_end[snd["ping"], snd["beam"], :3] = to_td(xyz_sl, z_ref).astype(np.float32)
    beam_end[snd["ping"], snd["beam"], 3] = 1.0
    origin_td = to_td(Ts[:, :3, 3], z_ref)
    Tv_sl = sl["T_world_vehicle_slam"] if slam_ok else snd["T_world_vehicle"]
    versions = {"K": sl["version_K"], "t_end": sl["version_t"], "n_loops": sl["version_loops"],
                "pos": sl["version_T"][:, :, :3, 3]}
    loops = [(int(i), int(j), float(t), sl["kf_T_dr"][i][:3, 3], sl["kf_T_dr"][j][:3, 3], float(r))
             for i, j, t, r in zip(sl["loop_i"], sl["loop_j"], sl["loop_t"], sl["loop_rmse"])]
    # loop endpoints: use corrected keyframe positions
    loops = [(i, j, t, sl["kf_T_opt"][i][:3, 3], sl["kf_T_opt"][j][:3, 3], r) for i, j, t, _, _, r in loops]
    truth_ping_pos = np.stack([np.interp(snd["ping_t"], tr["t"], tr["p"][:, k]) for k in range(3)], -1)
    d = {"z_ref": z_ref, "ping_t": snd["ping_t"], "display_idx": idx, "xyz_dr": xyz_dr, "xyz_slam": xyz_sl,
         "intensity": snd["intensity"], "rng": snd["rng"], "t": snd["t"], "sigma_h": snd["sigma_h"], "ping": snd["ping"],
         "ping_pos_dr": snd["T_world_vehicle"][:, :3, 3], "ping_pos_slam": Tv_sl[:, :3, 3], "ping_pos_truth": truth_ping_pos,
         "T_vehicle_slam": Tv_sl, "T_vehicle_dr": snd["T_world_vehicle"],
         "ping_sigma_dr": np.interp(snd["ping_t"], np.load(os.path.join(proc, "nav_dr.npz"))["t"], np.load(os.path.join(proc, "nav_dr.npz"))["sigma_xy"]),
         "beam_end_td": beam_end, "beam_origin_td": origin_td, "beam_theta": np.load(os.path.join(sim, "pings.npz"))["beam_theta"],
         "recon_tex": recon.astype(np.float32), "recon_extent": extent, "loops": loops, "versions": versions,
         "kf_t": sl["kf_t"], "candidates": cands, "truth_objects": meta_sim["objects"], "slam_available": slam_ok,
         "label": meta_sim["label"], "dataset": cfg["dataset"]["name"]}
    sr = json.load(open(os.path.join(out, "diagnostics", "slam_report.json")))
    d["slam_stats"] = {k: sr.get(k) for k in ("odometry_edges", "sonar_odometry_edges", "loop_candidates", "loop_accepted",
                                              "loop_accepted_after_screening", "loops_screened_out", "slam_available")}
    rx0, rx1 = (float(extent[0]), float(extent[1]))
    ry0, ry1 = (float(extent[2]), float(extent[3]))
    nxy = recon.shape[0]
    cx = int(max(0, np.floor((-hx - rx0) / (rx1 - rx0) * nxy))), int(min(nxy, np.ceil((hx - rx0) / (rx1 - rx0) * nxy)))
    cy = int(max(0, np.floor((-hy - ry0) / (ry1 - ry0) * nxy))), int(min(nxy, np.ceil((hy - ry0) / (ry1 - ry0) * nxy)))
    d["planned_cells"] = (cx[1] - cx[0]) * (cy[1] - cy[0])
    meta = export_all(os.path.join(out, "pointclouds"), d)
    rep = {"slam_used": slam_ok, "grid_spacing_m": sp, "grid_shape": list(g_sl.z.shape),
           "survey_coverage_fraction": cov, "soundings_used": int(len(xyz_sl)), "evaluation_vs_truth": ev,
           "uncertainty": um.summary(g_sl), "obstacles": {"candidates": len(cands), "evaluation": obs_eval},
           "occupancy": occ_stats, "contours": {"polylines": len(cs), "segments": int(len(seg)),
                                                "interval_m": mc["contour_interval_m"]},
           "interpolated_cells": int(g_sl.interpolated.sum()), "mesh": mmeta, "touchdesigner_export": meta,
           "terrain_depth_range_m": [float(np.nanmin(g_sl.z)), float(np.nanmax(g_sl.z))]}
    with open(os.path.join(out, "diagnostics", "map_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    rep = build(load_config(ap.parse_args().config))
    print(json.dumps({k: rep[k] for k in ("slam_used", "survey_coverage_fraction", "evaluation_vs_truth", "obstacles",
                                          "occupancy")}, indent=1, default=float))
