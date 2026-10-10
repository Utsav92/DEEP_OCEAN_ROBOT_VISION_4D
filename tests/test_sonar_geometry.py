import numpy as np
import pytest

from preprocessing import beam_geometry as bg, coordinate_systems as cs
from preprocessing.sonar_loader import Pings, load_mblist_ascii
from preprocessing import sonar_filter as sf
from sonar import multibeam as mb, uncertainty as un, forward_looking as fls, sidescan as ss, range_bearing as rb


def test_planar_beam_projection_matches_formula():
    th = np.radians([-60.0, 0.0, 30.0])
    xyz = bg.multibeam_xyz(np.array([100.0, 50.0, 80.0]), th)
    assert np.allclose(xyz[:, 0], 0.0)                                   # flat array: no along-track component
    assert np.allclose(xyz[:, 1], np.array([100, 50, 80]) * np.sin(th))
    assert np.allclose(xyz[:, 2], np.array([100, 50, 80]) * np.cos(th))
    assert np.allclose(np.linalg.norm(xyz, axis=1), [100, 50, 80])


def test_nadir_beam_points_down_in_world():
    # vehicle level at depth 100 m (z=-100): a 30 m nadir return lies at z = -130 below the sonar
    Tv = cs.pose_enu([0, 0, -100.0], 0, 0, 0.3)
    xs = bg.multibeam_xyz(np.array([[30.0]]), np.array([0.0]))
    Pw = mb.world_points(xs, Tv[None] @ np.eye(4)[None])
    assert Pw[0, 0, 2] == pytest.approx(-130.0)
    assert np.allclose(Pw[0, 0, :2], 0.0)


def test_starboard_beam_goes_right_of_heading():
    # heading north: starboard (positive theta) must be East of the track
    Tv = cs.pose_enu([0, 0, -100.0], 0, 0, 0.0)
    xs = bg.multibeam_xyz(np.array([[50.0]]), np.array([np.radians(45.0)]))
    Pw = mb.world_points(xs, Tv[None])
    assert Pw[0, 0, 0] > 30 and abs(Pw[0, 0, 1]) < 1e-9


def test_sensor_extrinsics_lever_arm_and_mount():
    Tv = cs.pose_enu([100, 200, -50.0], 0, 0, np.pi / 2)         # heading east
    Tvs = cs.mount_transform([2.0, 0.0, 0.5], [0, 0, 0])
    Ts = Tv @ Tvs
    assert np.allclose(Ts[:3, 3], [102.0, 200.0, -50.5])           # 2 m forward (east), 0.5 m down


def test_attitude_rotates_the_swath():
    # positive roll = starboard side DOWN; the belly (body +z) then faces PORT, so a nadir beam swings to the west
    Tv = cs.pose_enu([0, 0, -100.0], np.radians(10), 0, 0)
    Pw = mb.world_points(bg.multibeam_xyz(np.array([[50.0]]), np.array([0.0])), Tv[None])
    assert Pw[0, 0, 0] == pytest.approx(-50 * np.sin(np.radians(10)), rel=1e-6)


def test_invalid_measurement_rejection():
    r = np.array([[np.nan, 2.0, 50.0, 400.0, 100.0]])
    ok = sf.invalid_range_mask(r, 3.0, 260.0)
    assert ok.tolist() == [[False, False, True, False, True]]
    q = np.array([[0, 1, 2, 3, 3]])
    assert sf.quality_mask(q, 2).tolist() == [[False, False, True, True, True]]


def test_beam_outlier_filter_removes_spike():
    z = np.tile(np.linspace(80, 90, 41), (3, 1))
    z[1, 20] = 20.0                                              # spurious early return
    keep = sf.beam_outlier_mask(z, np.ones_like(z, bool))
    assert not keep[1, 20] and keep.sum() == z.size - 1


def test_uncertainty_propagation_first_order():
    sh, sv = un.sounding_sigma(np.array([100.0]), np.array([0.0]), 0.2, np.radians(0.2))
    assert sv[0] == pytest.approx(0.2, rel=1e-6)                  # nadir: depth error = range error
    assert sh[0] == pytest.approx(100 * np.radians(0.2), rel=1e-6)  # across-track error = r * sigma_theta
    sh2, _ = un.sounding_sigma(np.array([100.0]), np.array([0.0]), 0.2, np.radians(0.2), sigma_nav_xy=3.0)
    assert sh2[0] > sh[0]


def test_fls_polar_to_cartesian_roundtrip_and_no_depth():
    nr, nb = 64, 33
    b = np.radians(np.linspace(-60, 60, nb))
    img = np.zeros((nr, nb))
    img[40, 16] = 1.0                                             # a single bright return at ~range bin 40, bearing 0
    cart, valid, X, Y = fls.to_cartesian(img, 80.0, b, size=128)
    iy, ix = np.unravel_index(np.argmax(cart), cart.shape)
    assert abs(X[iy, ix] - (40.5 / 64 * 80.0)) < 2.0 and abs(Y[iy, ix]) < 2.0
    assert valid.sum() > 0 and not valid[0, 0]                    # corners outside the fan are invalid


def test_side_scan_has_no_height_conversion():
    g = ss.slant_to_ground(np.array([5.0, 10.0, 50.0]), altitude=10.0)
    assert np.isnan(g[0]) and np.isnan(g[1])
    assert g[2] == pytest.approx(np.sqrt(50 ** 2 - 10 ** 2))
    assert not hasattr(ss, "intensity_to_height")


def test_mblist_adapter_parses_and_flags_no_geometry():
    txt = "# comment\n1700000000.0 -122.0 36.77 850.2 0.5\n1700000001.0 -122.0001 36.7701 851.0 0.4\nbad line\n"
    d = load_mblist_ascii(txt)
    assert len(d["t"]) == 2 and d["geometry_available"] is False
    assert np.allclose(d["elevation_m"], [-850.2, -851.0])


def test_range_bearing_inverse():
    x, y = rb.polar_to_cart(np.array([10.0]), np.array([np.radians(30.0)]))
    r, b = rb.cart_to_polar(x, y)
    assert r[0] == pytest.approx(10.0) and b[0] == pytest.approx(np.radians(30.0))
