"""Source A-E inspection: which data portals are reachable and which processing tools are installed here.
Writes output/diagnostics/sources.json. Nothing is downloaded; a failed probe is recorded, never hidden."""
import json
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath, ensure_dir  # noqa: E402

SOURCES = [
    ("A", "GEBCO gridded bathymetry", "https://www.gebco.net/",
     "Global NetCDF grid; read with preprocessing.bathymetry_loader.load_netcdf (elevation variable, lat/lon)."),
    ("B", "NOAA NCEI bathymetry (CUDEM / multibeam)", "https://www.ncei.noaa.gov/maps/bathymetry/",
     "Used: DEM global mosaic ImageServer export (GeoTIFF, EPSG:4326, 1/9 arc-second CUDEM tiles where available)."),
    ("C", "MB-System", "https://www.mbari.org/technology/mb-system/",
     "Command-line multibeam processing; needs `mbinfo`/`mblist` on PATH and a supported raw format."),
    ("C", "MB-System repository", "https://github.com/dwcaress/MB-System", "Source and format list."),
    ("D", "MBARI", "https://www.mbari.org/", "Underwater robotics datasets (verify sensor channels per dataset)."),
    ("D", "Woods Hole Oceanographic Institution", "https://www.whoi.edu/", "Research datasets (verify per dataset)."),
    ("D", "University of Girona underwater robotics", "https://cirs.udg.edu/", "Sonar SLAM datasets (verify per dataset)."),
    ("E", "Stonefish simulator", "https://github.com/patrykcieslak/stonefish", "Underwater simulator (not installed)."),
]


def probe(url, timeout=12):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dorv-inspect/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"reachable": True, "status": r.status}
    except Exception as e:
        return {"reachable": False, "error": str(e)[:160]}


def tools():
    out = {}
    for t in ("mbinfo", "mblist", "ffmpeg", "gdalinfo", "ros2"):
        out[t] = shutil.which(t)
    for mod in ("gtsam", "torch", "rasterio", "netCDF4", "pyproj", "skimage", "cv2", "rclpy"):
        try:
            __import__(mod)
            out[mod] = "installed"
        except Exception:
            out[mod] = None
    return out


def main(cfg_path=None):
    cfg = load_config(cfg_path)
    rep = {"sources": [], "tools": tools(), "selected": {
        "real_bathymetry": cfg["dataset"]["name"], "sonar": "SIMULATED multibeam + forward-looking sonar (Source E fallback)",
        "reason": "no raw multibeam/sonar sequence with calibration + navigation was verified as downloadable here"}}
    for tag, name, url, note in SOURCES:
        rep["sources"].append({"tag": tag, "name": name, "url": url, "note": note, **probe(url)})
    d = ensure_dir(os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics"))
    with open(os.path.join(d, "sources.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else None)
    for s in r["sources"]:
        print(f"[{s['tag']}] {s['name']:42s} {'OK' if s['reachable'] else 'UNREACHABLE'}")
    print({k: v for k, v in r["tools"].items()})
