import numpy as np
import pytest

from navigation import depth_sensor as ds, dvl, imu, pose_interpolation as pi, state_estimator as se
from preprocessing import coordinate_systems as cs, timestamp_sync as ts, navigation_loader as nl, calibration as cal


def test_depth_from_pressure_unesco_check():
    # UNESCO 1983 check value: depth(P=10000 dbar, lat 30) = 9712.653 m
    assert ds.depth_from_pressure(10000.0, 30.0) == pytest.approx(9712.653, abs=0.01)


def test_pressure_depth_roundtrip():
    d = np.array([10.0, 565.0, 1031.0])
    assert np.allclose(ds.depth_from_pressure(ds.pressure_from_depth(d, 36.77), 36.77), d, atol=1e-3)


def test_dead_reckoning_straight_line_and_depth():
    t = np.arange(0, 101, 1.0)
    n = len(t)
    yaw = np.full(n, np.pi / 2)                                   # heading east
    v = np.tile([2.0, 0.0, 0.0], (n, 1))
    pos, ex = dvl.dead_reckon(t, np.zeros(n), np.zeros(n), yaw, v, np.ones(n, bool), np.full(n, 700.0))
    assert pos[-1, 0] == pytest.approx(200.0) and abs(pos[-1, 1]) < 1e-9
    assert np.allclose(pos[:, 2], -700.0) and not ex.any()


def test_dvl_dropout_holds_velocity_and_is_flagged():
    t = np.arange(0, 11, 1.0)
    n = len(t)
    v = np.tile([1.0, 0.0, 0.0], (n, 1))
    valid = np.ones(n, bool)
    valid[4:7] = False
    v[4:7] = np.nan
    pos, ex = dvl.dead_reckon(t, np.zeros(n), np.zeros(n), np.zeros(n), v, valid, np.zeros(n))
    assert ex.sum() == 3
    assert pos[-1, 1] == pytest.approx(10.0)                      # heading north at 1 m/s through the gap
    assert np.all(np.isfinite(pos))


def test_dvl_model_dropouts_marked_nan():
    m = dvl.DvlModel(dropouts=[(5.0, 8.0)])
    t = np.arange(0, 12.0)
    v, ok = m.measure(t, np.ones((12, 3)), np.random.default_rng(0))
    assert (~ok).sum() == 4 and np.isnan(v[~ok]).all()


def test_pose_interpolation_and_flags():
    t = np.array([0.0, 10.0])
    p = np.array([[0, 0, 0], [100, 0, 0.0]])
    R = np.stack([cs.R_enu_body(0, 0, 0), cs.R_enu_body(0, 0, np.pi / 2)])
    tr = pi.PoseTrack(t, p, R)
    T, st = tr.at([5.0, -1.0, 12.0])
    assert np.allclose(T[0, :3, 3], [50, 0, 0])
    assert cs.rpy_from_R_enu_body(T[0, :3, :3])[2] == pytest.approx(np.pi / 4)
    assert st.tolist() == [0, 1, 1]


def test_pose_track_rejects_unordered_time():
    with pytest.raises(ValueError):
        pi.PoseTrack([0, 2, 1], np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)))


def test_sort_and_dedupe_timestamps():
    t = np.array([3.0, 1.0, 2.0, 2.0])
    out = ts.sort_and_dedupe(t, np.array([30, 10, 20, 21]))
    assert out[0].tolist() == [1.0, 2.0, 3.0]
    assert ts.check_monotonic(out[0])["monotonic"]


def test_clock_offset_estimation():
    nav_t = np.arange(0, 200, 0.2)
    sig = np.sin(nav_t / 7.0) + 0.3 * np.sin(nav_t / 2.3)
    ping_t = np.arange(10, 150, 1.0)
    true_off = 0.37
    ping_sig = np.interp(ping_t + true_off, nav_t, sig)
    off, c = ts.estimate_clock_offset(ping_t, ping_sig, nav_t, sig, search_s=1.0, step_s=0.01)
    assert off == pytest.approx(true_off, abs=0.02) and c > 0.99


def test_nav_loader_requires_channels(tmp_path):
    p = tmp_path / "n.npz"
    np.savez(p, t=np.arange(3.0), roll=np.zeros(3))
    with pytest.raises(ValueError):
        nl.load_nav_npz(str(p))


def test_calibration_validation():
    assert cal.Calibration(cs.mount_transform([0.5, 0, 0.3], [0, 0, 0]), "config").validate()
    bad = np.eye(4)
    bad[0, 3] = 100.0
    with pytest.raises(ValueError):
        cal.Calibration(bad, "config").validate()
    assert cal.uncalibrated().calibrated is False


def test_state_estimator_uncertainty_grows():
    t = np.arange(0, 301, 1.0)
    n = len(t)
    nav = {"t": t, "roll": np.zeros(n), "pitch": np.zeros(n), "yaw": np.zeros(n), "v_body": np.tile([2.0, 0, 0], (n, 1)),
           "dvl_valid": np.ones(n, bool), "depth": np.full(n, 100.0)}
    est = se.estimate(nav, {"dvl_scale_error": 0.003})
    assert est["sigma_xy"][-1] > est["sigma_xy"][0]
    assert est["distance"][-1] == pytest.approx(600.0)
