import numpy as np
import pytest
from pyproj import Geod

from preprocessing import coordinate_systems as cs


def test_local_frame_metric_distance_matches_geodesic():
    f = cs.LocalFrame(-122.0, 36.77)
    lon = np.array([-122.0, -121.95, -122.0, -121.97])
    lat = np.array([36.77, 36.77, 36.82, 36.74])
    x, y = f.to_local(lon, lat)
    geod = Geod(ellps="WGS84")
    for k in (1, 2, 3):
        _, _, d = geod.inv(lon[0], lat[0], lon[k], lat[k])
        assert np.hypot(x[k] - x[0], y[k] - y[0]) == pytest.approx(d, rel=2e-4)


def test_one_degree_is_not_a_metre():
    f = cs.LocalFrame(-122.0, 36.77)
    x, _ = f.to_local([-121.99], [36.77])
    assert 850 < x[0] < 930                                    # 0.01 deg of lon ~ 890 m at 36.8 N, never 0.01


def test_roundtrip_geo_local():
    f = cs.LocalFrame(-122.0, 36.77)
    lon, lat = np.array([-122.012, -121.99]), np.array([36.76, 36.79])
    x, y = f.to_local(lon, lat)
    lon2, lat2 = f.to_geo(x, y)
    assert np.allclose(lon, lon2, atol=1e-9) and np.allclose(lat, lat2, atol=1e-9)


def test_body_axes_ned_convention():
    # level, heading north: body forward -> +North(ENU y), starboard -> +East, down -> -Up
    R = cs.R_enu_body(0.0, 0.0, 0.0)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(R @ [0, 1, 0], [1, 0, 0])
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1])
    # heading east (yaw 90 deg): forward -> +East
    assert np.allclose(cs.R_enu_body(0, 0, np.pi / 2) @ [1, 0, 0], [1, 0, 0], atol=1e-12)
    # pitch up 30 deg heading north: forward gains +Up
    v = cs.R_enu_body(0, np.radians(30), 0) @ [1, 0, 0]
    assert v[2] == pytest.approx(np.sin(np.radians(30)))


def test_rpy_roundtrip():
    rpy = (0.1, -0.2, 2.5)
    R = cs.R_enu_body(*rpy)
    assert np.allclose(cs.rpy_from_R_enu_body(R), rpy)


def test_pose_composition_and_inverse():
    T1 = cs.pose_enu([10, 20, -30], 0.1, 0.05, 1.0)
    T2 = cs.mount_transform([0.6, 0.0, 0.35], [0, 10, 5])
    T = cs.compose(T1, T2)
    p = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(cs.transform_points(T, p), cs.transform_points(T1, cs.transform_points(T2, p)))
    assert np.allclose(cs.compose(T, cs.invert(T)), np.eye(4), atol=1e-12)


def test_td_axes_roundtrip_and_handedness():
    p = np.array([[100.0, 200.0, -50.0]])
    q = cs.enu_to_td(p)
    assert np.allclose(q, [[100.0, -50.0, -200.0]])          # x=E, y=U, z=-N
    assert np.allclose(cs.td_to_enu(q), p)
    assert np.linalg.det(cs.T_td_enu()[:3, :3]) == pytest.approx(1.0)


def test_interpolate_pose_midpoint():
    A = cs.pose_enu([0, 0, 0], 0, 0, 0)
    B = cs.pose_enu([10, 0, 0], 0, 0, np.radians(90))
    M = cs.interpolate_pose(A, B, 0.5)
    assert np.allclose(M[:3, 3], [5, 0, 0])
    assert cs.rpy_from_R_enu_body(M[:3, :3])[2] == pytest.approx(np.radians(45), abs=1e-9)
