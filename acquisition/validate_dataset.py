"""Dataset integrity checks: file existence, format validation and metadata completeness."""
import json
import os

import numpy as np

REQUIRED_MANIFEST_KEYS = ("dataset_name", "source_url", "dataset_version", "downloaded_files", "file_hashes", "license",
                          "sensor_modalities", "coordinate_reference_system", "time_base", "calibration_available",
                          "navigation_available", "measurement_units", "supported_processing_modes", "missing_requirements")


def check_files(paths):
    out = []
    for p in paths:
        ok = os.path.isfile(p) and os.path.getsize(p) > 0
        out.append({"file": p, "exists": ok, "bytes": os.path.getsize(p) if ok else 0})
    return out


def check_geotiff(path):
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1, masked=True)
        return {"readable": True, "crs": str(ds.crs), "shape": list(a.shape), "has_data": bool(a.count() > 0),
                "float_elevation": str(ds.dtypes[0]).startswith("float")}


def check_npz(path, keys):
    d = np.load(path, allow_pickle=False)
    missing = [k for k in keys if k not in d.files]
    return {"readable": True, "missing_keys": missing}


def check_manifest(manifest):
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    empty = [k for k in REQUIRED_MANIFEST_KEYS if k in manifest and manifest[k] in (None, "", [], {})
             and k not in ("missing_requirements",)]
    return {"complete": not missing and not empty, "missing": missing, "empty": empty}


def check_all(root, manifest_path):
    m = json.load(open(manifest_path))
    return {"manifest": check_manifest(m),
            "files": check_files([os.path.join(root, f) for f in m.get("downloaded_files", [])])}
