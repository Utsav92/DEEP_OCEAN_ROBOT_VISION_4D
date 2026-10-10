"""Evaluate the auto-encoder: reconstruction error vs a mean-patch baseline, and cluster quality (silhouette)."""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath   # noqa: E402
from learning import sonar_dataset as ds, terrain_classifier as tc   # noqa: E402
from learning.sonar_encoder import PatchAutoEncoder     # noqa: E402
from learning.train import load_grid                     # noqa: E402


def evaluate(cfg, k=5):
    out = os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics")
    ck = torch.load(os.path.join(out, "patch_ae.pt"), weights_only=False)
    net = PatchAutoEncoder(32, 16)
    net.load_state_dict(ck["state"])
    net.eval()
    g = load_grid(cfg)
    P, pos = ds.extract_patches(g.z_interp, g.observed, 32, 8)
    X, _ = ds.normalise(P, ck["scale"])
    _, va = ds.split(len(X), 0.8, 0)
    xt = torch.tensor(X[:, None])
    with torch.no_grad():
        rec, z = net(xt)
    mse = float(((rec - xt) ** 2)[va].mean())
    base = float((xt[va] ** 2).mean())                    # predicting zero relief
    lab, _ = tc.kmeans(z.numpy(), k)
    sil = tc.silhouette(z.numpy(), lab)
    res = {"patches": int(len(X)), "val_mse": mse, "zero_relief_baseline_mse": base,
           "improvement_over_baseline": 1 - mse / max(base, 1e-9), "kmeans_k": k, "silhouette": sil,
           "note": "clusters are unlabelled terrain regimes, not semantic classes"}
    json.dump(res, open(os.path.join(out, "learning_eval.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(evaluate(load_config()), indent=1))
