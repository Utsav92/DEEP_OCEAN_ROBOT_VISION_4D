"""Inference: assign every observed 32x32 patch of the reconstructed seabed to an unlabelled terrain regime."""
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


def regimes(cfg, k=5):
    out = os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics")
    ck = torch.load(os.path.join(out, "patch_ae.pt"), weights_only=False)
    net = PatchAutoEncoder(32, 16)
    net.load_state_dict(ck["state"])
    net.eval()
    g = load_grid(cfg)
    P, pos = ds.extract_patches(g.z_interp, g.observed, 32, 16)
    X, _ = ds.normalise(P, ck["scale"])
    with torch.no_grad():
        z = net.encode(torch.tensor(X[:, None])).numpy()
    lab, _ = tc.kmeans(z, k)
    cmap = -np.ones(g.z.shape, np.int16)
    for (iy, ix), l in zip(pos, lab):
        cmap[iy:iy + 32, ix:ix + 32] = l
    np.save(os.path.join(out, "terrain_regimes.npy"), cmap)
    return {"patches": int(len(P)), "k": k, "counts": np.bincount(lab, minlength=k).tolist()}


if __name__ == "__main__":
    print(json.dumps(regimes(load_config()), indent=1))
