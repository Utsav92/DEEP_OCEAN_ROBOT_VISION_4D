import numpy as np
import pytest

from temporal_4d import sonar_history as sh, time_scrubber as ts, temporal_points as tp, time_explosion as te
from temporal_4d import scan_echoes as se, vehicle_history as vh, map_versions as mv, temporal_export as tx
from preprocessing import coordinate_systems as cs


def _history(P=100, per=5):
    t = np.arange(P) * 2.0
    off = np.arange(P + 1) * per
    xyz = np.column_stack([np.repeat(np.arange(P), per) * 5.0, np.tile(np.arange(per), P) * 1.0, np.full(P * per, -500.0)])
    return sh.SonarHistory(t, off, xyz, max_scans=50)


def test_history_ordering_and_window():
    h = _history()
    a, b = h.window(t=100.0, n=10)
    assert (a, b) == (41, 50)
    pts, ages, ping = h.points(100.0, 10)
    assert len(pts) == 10 * 5 and np.all(np.diff(ping) >= 0)
    assert ages.min() == 0.0 and ages.max() == pytest.approx(18.0)


def test_history_bounded_by_max_scans():
    h = _history()
    a, b = h.window(190.0, 500)
    assert b - a + 1 == 50


def test_history_before_first_scan_is_empty():
    h = _history()
    assert h.window(-5.0, 10) == (0, -1)
    assert len(h.points(-5.0, 10)[0]) == 0


def test_world_space_stability_under_scrubbing():
    h = _history()
    p1 = h.points(60.0, 20)[0].copy()
    h.points(150.0, 20)
    h.points(10.0, 20)
    p2 = h.points(60.0, 20)[0]
    assert np.array_equal(p1, p2)                              # history does not depend on playback order/camera


def test_scrubber_reverse_step_freeze():
    s = ts.TimeScrubber(np.arange(0, 200, 2.0), speed=10.0)
    s.set_time(100.0)
    s.reverse()
    t0 = s.t
    s.tick(1.0)
    assert s.t == pytest.approx(t0 - 10.0)
    s.pause()
    assert s.step(+3) == s.scan_index() and s.t == pytest.approx(s.ping_t[s.scan_index()])
    s.play()
    s.freeze(True)
    tf = s.t
    s.tick(5.0)
    assert s.t == tf                                           # frozen time does not advance (camera is separate)
    s.freeze(False)
    s.tick(1.0)
    assert s.t > tf


def test_scrubber_jumps_and_loop():
    s = ts.TimeScrubber(np.arange(0, 100, 2.0), version_t=[10.0, 50.0, 90.0], speed=100.0)
    assert s.jump_to_scan(7) == 7 and s.t == 14.0
    assert s.jump_to_version(1) == 1 and s.t == 50.0
    s.play()
    s.tick(2.0)                                                 # +200 s wraps around the 98 s survey
    assert s.t0 <= s.t <= s.t1


def test_timestamp_interpolation_of_vehicle_pose():
    t = np.array([0.0, 10.0])
    T = np.stack([cs.pose_enu([0, 0, -100], 0, 0, 0), cs.pose_enu([100, 0, -100], 0, 0, 0)])
    h = vh.VehicleHistory(t, T)
    assert np.allclose(h.pose_at(2.5)[:3, 3], [25, 0, -100])
    assert np.allclose(h.pose_at(99.0)[:3, 3], [100, 0, -100])   # clamped, never extrapolated


def test_opacity_decay_and_visibility():
    assert tp.opacity(0.0, 30.0) == 1.0
    assert tp.opacity(30.0, 30.0) == pytest.approx(np.exp(-1))
    assert tp.opacity(-5.0, 30.0) == 0.0                         # not yet observed
    m = tp.visible_mask(np.array([0, 50, 90, 120.0]), 100.0, 60.0)
    assert m.tolist() == [False, True, True, False]      # ages 100, 50, 10, -20 vs horizon 60


def test_time_explosion_and_echo_are_stylised_functions_of_age():
    d = te.displacement(np.array([0.0, 10.0, 1e5]), 2.0, "up", clamp_m=500.0)
    assert np.allclose(d[:, 2], [0, 20, 500]) and np.allclose(d[:, :2], 0)
    assert se.echo_alpha(np.array([0.0, 2.0, 5.0])).tolist() == [1.0, 0.25, 0.0]


def test_map_versions_roundtrip(tmp_path):
    K = np.array([3, 5])
    T = np.tile(np.eye(4), (2, 5, 1, 1))
    T[1, :, 0, 3] = np.arange(5)
    kf_T = np.tile(np.eye(4), (5, 1, 1))
    m = mv.MapVersions(K, [10.0, 20.0], [0, 4], T, kf_T, np.arange(5, dtype=float) * 4)
    assert m.version_at_time(15.0) == 0 and m.version_at_time(25.0) == 1
    p, t = m.trajectory(1)
    assert len(p) == 5 and m.correction_magnitude(1).max() == pytest.approx(4.0)


def test_export_td_axes_and_zref():
    p = np.array([[10.0, 20.0, -780.0]])
    q = tx.to_td(p, -800.0)
    assert np.allclose(q, [[10.0, 20.0, -20.0]])                 # x=E, y=U-zref, z=-N
    tex = tx.path_texture(np.zeros((3, 3)), np.array([0.0, 1.0, 2.0]), 4)
    assert tex.shape == (4, 4, 4) and tex[0, 0, 3] == 0.0 and tex[3, 3, 3] == 1e9
    P, T = tx.densify(np.array([[0.0, 0, 0], [10.0, 0, 0]]), np.array([0.0, 4.0]), 4)
    assert len(P) == 5 and np.all(np.diff(T) >= 0)
