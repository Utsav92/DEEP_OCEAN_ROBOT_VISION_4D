"""Inspect a gridded bathymetry file (GeoTIFF or GEBCO-style NetCDF): CRS, extent, cell size, NoData, depth range."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing import bathymetry_loader as bl  # noqa: E402


def inspect(path):
    ext = os.path.splitext(path)[1].lower()
    g = bl.load_netcdf(path) if ext in (".nc", ".nc4") else bl.load_geotiff(path)
    z = g.z
    lon_c = float(np.mean(np.diff(g.lon)))
    lat_c = float(np.mean(np.diff(g.lat)))
    return {"file": os.path.basename(path), "format": "NetCDF" if ext in (".nc", ".nc4") else "GeoTIFF",
            "crs": g.crs, "shape": list(z.shape), "bbox_lon_lat": list(g.bbox),
            "cell_deg": [abs(lon_c), abs(lat_c)],
            "cell_m_approx": [abs(lon_c) * 111320 * np.cos(np.radians(np.mean(g.lat))), abs(lat_c) * 110574],
            "valid_fraction": g.valid_fraction(), "elevation_min_m": float(np.nanmin(z)), "elevation_max_m": float(np.nanmax(z)),
            "vertical_convention": "elevation m, up positive (depth = -elevation)", "vertical_datum": g.vertical_datum}


if __name__ == "__main__":
    print(json.dumps(inspect(sys.argv[1]), indent=1))
