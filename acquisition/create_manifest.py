"""Create output/diagnostics/dataset_manifest.json: provenance, hashes, licence, modalities, frames and what is missing."""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath, ensure_dir   # noqa: E402
from acquisition import validate_license, validate_dataset          # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(cfg):
    ds = cfg["dataset"]
    prov_path = os.path.join(ROOT, "data", "raw", "download_provenance.json")
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {}
    files = [ds["overview_file"], ds["survey_file"]]
    tiles = {}
    for k, v in prov.items():
        for t in v.get("source_tiles", []):
            if "Name" in t and t.get("Category") == 1:
                tiles[t["Name"]] = {"cell_arcsec": t.get("CellsizeArcseconds"), "vertical_datum": t.get("VerticalDatum"),
                                    "metadata_url": t.get("MetadataURL")}
    lic = {"name": ds["name"], "license": ds["license"], "provider": ds["provider"],
           "license_url": "https://www.ncei.noaa.gov/about/data-policy", "restrictions": "none stated by NOAA NCEI; "
           "attribution to NOAA NCEI requested; verify the per-tile metadata_url before redistribution"}
    sim_meta_p = os.path.join(abspath(cfg, cfg["paths"]["simulation"]), "sim_meta.json")
    sim_meta = json.load(open(sim_meta_p)) if os.path.exists(sim_meta_p) else None
    man = {
        "dataset_name": ds["name"], "source_url": ds["service_url"], "viewer_url": ds["viewer_url"],
        "dataset_version": "NCEI CUDEM 2023v1 tiles (ncei19_*_2023v1) via DEM_global_mosaic ImageServer, retrieved "
                           "from the live service (no service-side version string published)",
        "downloaded_files": files,
        "file_hashes": {f: sha256(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))},
        "source_tiles": tiles,
        "license": lic,
        "license_review": validate_license.validate(lic),
        "sensor_modalities": {"real": ["gridded bathymetry (compiled multibeam/lidar, elevation grid)"],
                              "simulated": ["multibeam echosounder", "forward-looking sonar", "DVL", "AHRS", "pressure"]},
        "coordinate_reference_system": {"source": "EPSG:4326 (WGS84 geographic)",
                                        "working": "local transverse Mercator ENU, origin at the survey-tile centre"},
        "time_base": "none for the real bathymetry (a compiled grid has no acquisition timestamps); simulated survey "
                     "uses seconds since mission start",
        "calibration_available": {"real_data": False, "simulated_sonar": "sensor mount + beam geometry from config.yaml"},
        "navigation_available": {"real_data": False, "simulated": "DVL + AHRS + depth dead reckoning with modelled error"},
        "measurement_units": {"elevation": "m (up positive)", "range": "m", "angle": "rad", "time": "s", "velocity": "m/s"},
        "supported_processing_modes": ["MODE 1 real bathymetry (works)", "MODE 5 simulated AUV (works)",
                                      "MODE 2 multibeam survey (pipeline exercised on SIMULATED soundings only)",
                                      "MODE 3 forward-looking sonar (simulated images)",
                                      "MODE 4 underwater SLAM (simulated sequence with ground truth)",
                                      "MODE 6 live ROS 2 (code present, never run)"],
        "missing_requirements": ["no real raw multibeam / sonar sequence with calibration and navigation was obtained",
                                 "MB-System not installed (mbinfo/mblist absent)", "GTSAM not installed (own robust "
                                 "SE(3) pose-graph solver used)", "no GPU with 12 GB VRAM (integrated graphics only)"],
        "simulation": sim_meta,
        "synthetic_flags": "all sonar, navigation and inserted seabed objects are SIMULATED; only the terrain is real",
    }
    d = ensure_dir(os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics"))
    out = os.path.join(d, "dataset_manifest.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=1)
    man["_validation"] = validate_dataset.check_manifest(man)
    return man, out


if __name__ == "__main__":
    m, p = build(load_config(sys.argv[1] if len(sys.argv) > 1 else None))
    print("manifest ->", p, m["_validation"])
