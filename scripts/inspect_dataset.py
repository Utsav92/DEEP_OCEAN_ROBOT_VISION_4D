"""Stage 1: dataset verification - sources, real tile inspection, licence review and manifest.

    python scripts/inspect_dataset.py --config config.yaml
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath   # noqa: E402
from acquisition import inspect_sources, inspect_survey, create_manifest, validate_dataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--offline", action="store_true", help="skip network probes")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not a.offline:
        r = inspect_sources.main(a.config)
        for s in r["sources"]:
            print(f"[{s['tag']}] {s['name']:42s} {'OK' if s['reachable'] else 'UNREACHABLE'}")
    for key in ("survey_file", "overview_file"):
        p = abspath(cfg, cfg["dataset"][key])
        if os.path.exists(p):
            print(json.dumps(inspect_survey.inspect(p), indent=1))
        else:
            print("MISSING (run acquisition/download_bathymetry.py):", p)
    man, out = create_manifest.build(cfg)
    print("manifest:", out)
    print("licence review:", man["license_review"])
    print("manifest validation:", man["_validation"])
    print("missing requirements:")
    for m in man["missing_requirements"]:
        print("  -", m)
