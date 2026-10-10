"""Stage 6: underwater SLAM (keyframes, odometry, verified loop closures, pose-graph optimisation, reintegration).

    python scripts/run_slam.py --config config.yaml
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath   # noqa: E402
from slam.slam_pipeline import run                      # noqa: E402


def evaluate(cfg):
    """Ground-truth evaluation (simulation only): absolute trajectory error of keyframes before/after SLAM."""
    proc, sim = abspath(cfg, cfg["paths"]["processed"]), abspath(cfg, cfg["paths"]["simulation"])
    r = np.load(os.path.join(proc, "slam_result.npz"))
    tr = np.load(os.path.join(sim, "truth.npz"))
    pt = np.stack([np.interp(r["kf_t"], tr["t"], tr["p"][:, k]) for k in range(3)], -1)
    e0 = np.linalg.norm(r["kf_T_dr"][:, :3, 3] - pt, axis=1)
    e1 = np.linalg.norm(r["kf_T_opt"][:, :3, 3] - pt, axis=1)
    out = {"ate_rmse_dr_m": float(np.sqrt(np.mean(e0 ** 2))), "ate_rmse_slam_m": float(np.sqrt(np.mean(e1 ** 2))),
           "ate_max_dr_m": float(e0.max()), "ate_max_slam_m": float(e1.max()),
           "ate_mean_dr_m": float(e0.mean()), "ate_mean_slam_m": float(e1.mean()), "evaluated_against": "simulated truth"}
    with open(os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics", "slam_evaluation.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--reuse-loops", action="store_true", help="reuse cached loop detection")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rep = run(cfg, reuse_loops=a.reuse_loops)
    print(json.dumps({k: v for k, v in rep.items() if k != "versions"}, indent=1, default=float))
    print(json.dumps(evaluate(cfg), indent=1))
