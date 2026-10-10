"""Stage 4/5: sonar + navigation preprocessing -> filtered world soundings (dead-reckoned).

    python scripts/preprocess_sonar.py --config config.yaml
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config          # noqa: E402
from preprocessing.process_sequence import preprocess  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    ss, est, rep = preprocess(cfg)
    print(json.dumps(rep["filter_stats"], indent=1))
    print("DR final sigma_xy %.1f m, DR distance %.0f m" % (est["sigma_xy"][-1], est["distance"][-1]))
