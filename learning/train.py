"""Train the relief-patch auto-encoder on the reconstructed seabed (CPU, offline).

    python learning/train.py --config config.yaml [--epochs 12]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing.config import load_config, abspath    # noqa: E402
from learning import sonar_dataset as ds                  # noqa: E402
from learning.sonar_encoder import PatchAutoEncoder       # noqa: E402


def load_grid(cfg):
    from mapping.bathymetry_grid import grid_soundings
    proc = abspath(cfg, cfg["paths"]["processed"])
    snd = np.load(os.path.join(proc, "soundings_dr.npz"))
    sl = np.load(os.path.join(proc, "slam_result.npz"))
    tt = np.load(os.path.join(abspath(cfg, cfg["paths"]["simulation"]), "truth_terrain.npz"))
    ext = (float(tt["x"][0]), float(tt["x"][-1]), float(tt["y"][0]), float(tt["y"][-1]))
    g = grid_soundings(sl["xyz_slam"].astype(float), snd["t"], snd["intensity"], snd["sigma_v"], snd["sigma_h"],
                       cfg["mapping"]["grid_spacing_m"], ext)
    return g


def main(cfg, epochs=12, seed=0):
    torch.manual_seed(seed)
    g = load_grid(cfg)
    P, pos = ds.extract_patches(g.z_interp, g.observed, 32, 8)
    X, scale = ds.normalise(P)
    tr, va = ds.split(len(X), 0.8, seed)
    X = torch.tensor(X[:, None])
    net = PatchAutoEncoder(32, 16)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    hist = []
    for ep in range(epochs):
        perm = torch.tensor(np.random.default_rng(ep).permutation(tr))
        for b in range(0, len(perm), 64):
            xb = X[perm[b:b + 64]]
            rec, _ = net(xb)
            loss = torch.nn.functional.mse_loss(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(net(X[va])[0], X[va]).item()
        hist.append(v)
        print(f"epoch {ep + 1:2d} val mse {v:.4f}")
    out = os.path.join(abspath(cfg, cfg["paths"]["output"]), "diagnostics")
    torch.save({"state": net.state_dict(), "scale": scale}, os.path.join(out, "patch_ae.pt"))
    json.dump({"patches": int(len(X)), "scale_m": scale, "val_mse": hist, "trained_on": "SIMULATED survey of real terrain"},
              open(os.path.join(out, "learning_train.json"), "w"), indent=1)
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    main(load_config(a.config), a.epochs)
