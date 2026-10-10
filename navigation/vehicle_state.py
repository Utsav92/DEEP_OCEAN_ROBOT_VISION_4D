"""Vehicle state record."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleState:
    t: float
    T_enu_vehicle: np.ndarray                 # 4x4, vehicle FRD -> local ENU
    velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sigma_xy_m: float = 0.0                   # 1-sigma horizontal position uncertainty of the estimate
    sigma_yaw_rad: float = 0.0
    depth_m: float = 0.0
    status: str = "ok"                        # ok | dvl_extrapolated | clamped | missing

    @property
    def position(self):
        return self.T_enu_vehicle[:3, 3]
