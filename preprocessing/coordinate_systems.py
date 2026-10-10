"""Coordinate frames and SE(3) utilities.

FRAMES (all right-handed)
  GEOGRAPHIC   WGS84 lon/lat (deg) + elevation (m, up positive, 0 = vertical datum of the source).
  LOCAL ENU    "world" frame of this project: transverse Mercator (k=1, WGS84) centred on the survey origin;
               x = East, y = North (grid north), z = Up. Sea surface ~ z = 0, seabed z < 0.
  NED          local north-east-down, used only for attitude conventions (roll/pitch/yaw), converted with R_ENU_NED.
  VEHICLE      body frame FRD: x forward, y starboard (right), z down.
  SONAR        sonar head frame, also FRD-aligned: x along-track (forward), y across-track (starboard), z down
               (boresight at nadir). Mounted on the vehicle with the extrinsic T_vehicle_sonar.
  TOUCHDESIGNER  world frame of the renderer: x = East, y = Up, z = -North (right-handed, y-up).

Poses are 4x4 homogeneous transforms T_a_b mapping coordinates in frame b to frame a (p_a = T_a_b p_b).
Attitude: R_NED_BODY = Rz(yaw) Ry(pitch) Rx(roll) (aerospace ZYX, yaw clockwise from north); R_ENU_BODY = R_ENU_NED R_NED_BODY.
"""
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation

# NED -> ENU: east = NED.y, north = NED.x, up = -NED.z (a 180 deg rotation about (1,1,0)/sqrt(2), det = +1)
R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


class LocalFrame:
    """Metric local frame: transverse Mercator on WGS84 centred at (lon0, lat0)."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.crs = CRS.from_proj4(f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} +k=1 +x_0=0 +y_0=0 "
                                  "+ellps=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def to_local(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return np.asarray(x), np.asarray(y)

    def to_geo(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return np.asarray(lon), np.asarray(lat)

    def meta(self):
        return {"type": "transverse_mercator_local", "lon0": self.lon0, "lat0": self.lat0, "k": 1.0,
                "ellipsoid": "WGS84", "axes": "x=East y=North z=Up", "proj4": self.crs.to_proj4()}


# ------------------------------------------------------------------ SE(3)
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def R_ned_body(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def R_enu_body(roll, pitch, yaw):
    return R_ENU_NED @ R_ned_body(roll, pitch, yaw)


def rpy_from_R_enu_body(R):
    """Inverse of R_enu_body -> (roll, pitch, yaw) in the NED aerospace convention."""
    Rn = R_ENU_NED.T @ R
    yaw = np.arctan2(Rn[1, 0], Rn[0, 0])
    pitch = -np.arcsin(np.clip(Rn[2, 0], -1.0, 1.0))
    roll = np.arctan2(Rn[2, 1], Rn[2, 2])
    return roll, pitch, yaw


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_enu(position_enu, roll, pitch, yaw):
    """T_enu_vehicle from an ENU position and NED-convention attitude (radians)."""
    return make_T(R_enu_body(roll, pitch, yaw), position_enu)


def invert(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(*Ts):
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def mount_transform(xyz, rpy_deg):
    """T_vehicle_sonar from a lever arm (FRD, m) and mounting angles (roll, pitch, yaw in deg, body-frame rotations)."""
    r, p, y = np.radians(rpy_deg)
    return make_T(rot_z(y) @ rot_y(p) @ rot_x(r), xyz)


# ------------------------------------------------------------------ Lie algebra helpers (rotation vector + translation)
def se3_log(T):
    """6-vector [translation(3); rotation vector(3)] of a pose (not the screw-motion log: translation is the plain t)."""
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def se3_exp(xi):
    return make_T(Rotation.from_rotvec(xi[3:]).as_matrix(), xi[:3])


def interpolate_pose(T0, T1, a):
    """Linear interpolation of position and spherical interpolation of rotation, a in [0,1]."""
    from scipy.spatial.transform import Slerp
    R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))([float(a)]).as_matrix()[0]
    return make_T(R, (1 - a) * T0[:3, 3] + a * T1[:3, 3])


# ------------------------------------------------------------------ TouchDesigner axes
def enu_to_td(p):
    """ENU (E,N,U) -> TouchDesigner (x=E, y=U, z=-N). Works on (..., 3) arrays."""
    p = np.asarray(p)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def td_to_enu(p):
    p = np.asarray(p)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def T_td_enu():
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    return T


def pose_enu_to_td(T_enu_vehicle):
    """Vehicle pose expressed in TouchDesigner axes (vehicle FRD body axes are kept as the object frame)."""
    return T_td_enu() @ T_enu_vehicle
