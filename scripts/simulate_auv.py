"""Mode 5: generate the simulated AUV survey (labelled SIMULATED) over the real bathymetry.

    python scripts/simulate_auv.py --config config.yaml
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath                    # noqa: E402
from preprocessing.bathymetry_loader import MetricGrid                    # noqa: E402
from preprocessing.coordinate_systems import LocalFrame                   # noqa: E402
from simulation.auv_sim import simulate                                   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    d = np.load(os.path.join(abspath(cfg, cfg["paths"]["processed"]), "bathymetry_survey.npz"))
    g = MetricGrid(d["x"], d["y"], d["z"].astype(np.float64), LocalFrame(float(d["lon0"]), float(d["lat0"])))
    t0 = time.time()
    meta = simulate(cfg, g, abspath(cfg, cfg["paths"]["simulation"]))
    print({k: v for k, v in meta.items() if k not in ("objects", "note")}, f"{time.time() - t0:.0f}s")
    for o in meta["objects"]:
        print("object", {k: round(v, 1) if isinstance(v, float) else v for k, v in o.items()})
