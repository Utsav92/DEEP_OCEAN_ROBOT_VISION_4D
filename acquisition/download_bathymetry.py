"""Download real gridded bathymetry from the NOAA NCEI DEM global mosaic ImageServer (Source B).

The mosaic serves NCEI CUDEM 1/9 arc-second tiles where they exist (this region) and coarser products elsewhere.
It returns a GeoTIFF (EPSG:4326, float32 elevation in metres, positive up). No synthetic data is created here.

    python acquisition/download_bathymetry.py --config config.yaml [--which overview|survey|both]
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath  # noqa: E402


def export_image(service_url, bbox, size, out_path, timeout=180):
    q = {"bbox": ",".join(f"{v:.8f}" for v in bbox), "bboxSR": "4326", "imageSR": "4326",
         "size": f"{size[0]},{size[1]}", "format": "tiff", "pixelType": "F32",
         "interpolation": "RSP_BilinearInterpolation", "noData": "-99999", "f": "image"}
    url = service_url + "/exportImage?" + urllib.parse.urlencode(q)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(out_path, "wb") as f:
        data = r.read()
        if data[:2] not in (b"II", b"MM"):
            raise RuntimeError("service did not return a TIFF: " + data[:200].decode("latin1"))
        f.write(data)
    return url


def source_tiles(service_url, bbox, timeout=60):
    """Catalog items (source DEM tiles) that intersect bbox: name, cell size, vertical datum, metadata URL."""
    q = {"geometry": ",".join(str(v) for v in bbox), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
         "spatialRel": "esriSpatialRelIntersects", "outFields": "Name,CellsizeArcseconds,VerticalDatum,MetadataURL,DateCompleted,Category",
         "returnGeometry": "false", "f": "json"}
    url = service_url + "/query?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        js = json.loads(r.read().decode("utf-8"))
    return [f["attributes"] for f in js.get("features", [])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--which", default="both", choices=["overview", "survey", "both"])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ds = cfg["dataset"]
    prov = {}
    jobs = []
    if a.which in ("overview", "both"):
        jobs.append(("overview", ds["overview_bbox"], ds["overview_pixels"], abspath(cfg, ds["overview_file"])))
    if a.which in ("survey", "both"):
        jobs.append(("survey", ds["survey_bbox"], ds["survey_pixels"], abspath(cfg, ds["survey_file"])))
    for name, bbox, size, out in jobs:
        print(f"[{name}] bbox={bbox} size={size} -> {out}")
        url = export_image(ds["service_url"], bbox, size, out)
        try:
            tiles = source_tiles(ds["service_url"], bbox)
        except Exception as e:  # provenance lookup is best-effort but its failure is recorded
            tiles = [{"error": str(e)}]
        prov[name] = {"request_url": url, "file": os.path.relpath(out, ROOT), "bbox": bbox, "size": size,
                      "source_tiles": tiles}
        print(f"[{name}] {os.path.getsize(out)} bytes, {len(tiles)} source catalog items")
    p = os.path.join(ROOT, "data", "raw", "download_provenance.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(prov)
    json.dump(old, open(p, "w"), indent=1)
    print("provenance ->", p)


if __name__ == "__main__":
    main()
