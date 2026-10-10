"""Sensor calibration container and validation."""
from dataclasses import dataclass

import numpy as np

from .coordinate_systems import mount_transform


@dataclass
class Calibration:
    T_vehicle_sonar: np.ndarray
    source: str                       # "file" | "config" | "default-identity (UNCALIBRATED)"
    calibrated: bool = True

    def validate(self):
        R = self.T_vehicle_sonar[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or abs(np.linalg.det(R) - 1) > 1e-6:
            raise ValueError("T_vehicle_sonar rotation is not a proper rotation")
        if np.linalg.norm(self.T_vehicle_sonar[:3, 3]) > 20.0:
            raise ValueError("sonar lever arm > 20 m: implausible for the vehicle")
        return True


def from_pings(pings):
    return Calibration(np.array(pings.T_vehicle_sonar, dtype=np.float64), "file")


def from_config(cfg_sonar):
    return Calibration(mount_transform(cfg_sonar["mount_xyz_m"], cfg_sonar["mount_rpy_deg"]), "config")


def uncalibrated():
    return Calibration(np.eye(4), "default-identity (UNCALIBRATED)", calibrated=False)
