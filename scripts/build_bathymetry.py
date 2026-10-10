"""Stage 2: real gridded bathymetry -> metric ENU grids, terrain meshes, height textures, diagnostics.

    python scripts/build_bathymetry.py --config config.yaml
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath, ensure_dir          # noqa: E402
from preprocessing.bathymetry_loader import load_geotiff, to_metric_grid     # noqa: E402
from preprocessing.coordinate_systems import LocalFrame                       # noqa: E402
from mapping.terrain_mesh import terrain_mesh, export_mesh_bundle             # noqa: E402
from mapping.map_export import resample_to_texture, save_texture              # noqa: E402
from mapping.surface_normals import slope_deg                                 # noqa: E402


def build(cfg):
    ds, bc = cfg["dataset"], cfg["bathymetry"]
    out = ensure_dir(abspath(cfg, cfg["paths"]["output"]))
    proc = ensure_dir(abspath(cfg, cfg["paths"]["processed"]))
    diag = ensure_dir(os.path.join(out, "diagnostics"))
    datum = "NAVD88 (NCEI CUDEM tile metadata)"
    sv = load_geotiff(abspath(cfg, ds["survey_file"]), bc["nodata_values"], bc["max_abs_elevation_m"], datum)
    ov = load_geotiff(abspath(cfg, ds["overview_file"]), bc["nodata_values"], bc["max_abs_elevation_m"], datum)
    lon0, lat0 = 0.5 * (sv.lon[0] + sv.lon[-1]), 0.5 * (sv.lat[0] + sv.lat[-1])
    frame = LocalFrame(lon0, lat0)
    msv = to_metric_grid(sv, frame, spacing=bc["grid_spacing_m"])
    mov = to_metric_grid(ov, frame, spacing=17.5)
    report = {"frame": frame.meta(), "vertical_convention": bc["vertical_convention"], "datum": datum}
    for name, g, geo in (("survey", msv, sv), ("overview", mov, ov)):
        np.savez_compressed(os.path.join(proc, f"bathymetry_{name}.npz"), x=g.x, y=g.y, z=g.z.astype(np.float32),
                            lon0=lon0, lat0=lat0)
        step = 2 if name == "survey" else 3
        mesh = terrain_mesh(g, step=step)
        meta = export_mesh_bundle(mesh, os.path.join(out, "meshes"), f"terrain_{name}",
                                  {"frame": "local ENU m", "source": geo.source, "datum": datum,
                                   "spacing_m": g.dx * step})
        size = bc["terrain_texture_size"] * (2 if name == "survey" else 1)
        tex, ok, ext = resample_to_texture(g.x, g.y, g.z, size)
        td = save_texture(os.path.join(out, "terrain", f"terrain_{name}"), tex, ext,
                          {"frame": frame.meta(), "datum": datum, "source": os.path.basename(geo.source),
                           "spacing_m": g.dx, "kind": "REAL gridded bathymetry (NOAA NCEI)"})
        sl = slope_deg(g.z, g.dx, g.dy)
        report[name] = {"shape": list(g.shape), "spacing_m": g.dx,
                        "extent_m": [float(g.x[0]), float(g.x[-1]), float(g.y[0]), float(g.y[-1])],
                        "elevation_min_m": float(np.nanmin(g.z)), "elevation_max_m": float(np.nanmax(g.z)),
                        "valid_fraction": float(np.isfinite(g.z).mean()),
                        "slope_deg_p50_p95_max": [float(np.nanpercentile(sl, 50)), float(np.nanpercentile(sl, 95)),
                                                  float(np.nanmax(sl))],
                        "mesh": meta, "texture": {"shape": td["shape"], "valid_fraction": td["valid_fraction"]}}
        print(f"[{name}] grid {g.shape} @ {g.dx:.2f} m, z {np.nanmin(g.z):.1f}..{np.nanmax(g.z):.1f} m, "
              f"mesh {meta['vertices']} verts / {meta['triangles']} tris")
    with open(os.path.join(diag, "bathymetry_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    build(load_config(ap.parse_args().config))
