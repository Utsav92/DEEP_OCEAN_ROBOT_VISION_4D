"""Trajectory container: poses over time with arc length, resampling and export."""
import json
import os

import numpy as np

from preprocessing.coordinate_systems import rpy_from_R_enu_body


class Trajectory:
    def __init__(self, t, positions, rotations, label="", meta=None):
        self.t = np.asarray(t, dtype=np.float64)
        self.p = np.asarray(positions, dtype=np.float64)
        self.R = np.asarray(rotations, dtype=np.float64)
        self.label = label
        self.meta = meta or {}

    @property
    def arclength(self):
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))])

    @property
    def length_m(self):
        return float(self.arclength[-1])

    def rpy_deg(self):
        return np.degrees(np.array([rpy_from_R_enu_body(R) for R in self.R]))

    def resample_by_distance(self, step_m):
        s = self.arclength
        sq = np.arange(0.0, s[-1], step_m)
        idx = np.searchsorted(s, sq, side="right") - 1
        return np.clip(idx, 0, len(s) - 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, t=self.t, p=self.p, R=self.R, label=self.label)
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump({"label": self.label, "n": int(len(self.t)), "length_m": self.length_m,
                       "t0": float(self.t[0]), "t1": float(self.t[-1]), **self.meta}, f, indent=1)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["t"], d["p"], d["R"], str(d["label"]))
