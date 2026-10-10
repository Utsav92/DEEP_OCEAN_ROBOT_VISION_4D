import numpy as np
import pytest

from learning import sonar_dataset as ds, terrain_classifier as tc
from optional_ros2 import sonar_bridge as sb, navigation_bridge as nb, vehicle_bridge as vb
from preprocessing import coordinate_systems as cs


def test_patches_skip_unobserved_and_remove_mean():
    z = np.random.default_rng(0).normal(size=(64, 64)) + 100.0
    obs = np.ones_like(z, bool)
    obs[:, :20] = False
    P, pos = ds.extract_patches(z, obs, 16, 8)
    assert len(P) > 0 and np.all(pos[:, 1] >= 20 - 7)
    assert np.allclose(P.mean(axis=(1, 2)), 0, atol=1e-4)


def test_kmeans_separates_two_regimes_and_features():
    rng = np.random.default_rng(0)
    flat = rng.normal(0, 0.05, (60, 16, 16))
    rough = rng.normal(0, 2.0, (60, 16, 16))
    X = tc.hand_features(np.concatenate([flat, rough]))
    lab, _ = tc.kmeans(X, 2)
    assert len(set(lab[:60])) == 1 and len(set(lab[60:])) == 1 and lab[0] != lab[-1]
    assert tc.silhouette(X, lab) > 0.5


def test_autoencoder_overfits_small_batch():
    torch = pytest.importorskip("torch")
    from learning.sonar_encoder import PatchAutoEncoder
    torch.manual_seed(0)
    x = torch.randn(16, 1, 32, 32).clamp(-1, 1) * 0.5
    net = PatchAutoEncoder(32, 16)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    l0 = torch.nn.functional.mse_loss(net(x)[0], x).item()
    for _ in range(80):
        loss = torch.nn.functional.mse_loss(net(x)[0], x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * l0


def test_ros_laserscan_conversion_marks_invalid():
    t, r, th, ok = sb.laserscan_to_ping([10.0, np.inf, 0.0, 400.0, 50.0], -0.5, 0.25, 12.5, range_max=260.0)
    assert ok.tolist() == [True, False, False, False, True]
    assert np.isnan(r[~ok]).all() and th[0] == -0.5 and t == 12.5


def test_ros_ned_odometry_to_project_frames():
    T = nb.odom_to_pose_enu(10.0, 20.0, 5.0, 0, 0, 0, 1, frame="ned")        # identity attitude in NED = level, north
    assert np.allclose(T[:3, 3], [20.0, 10.0, -5.0])                         # NED (N,E,D) -> ENU (E,N,U)
    assert np.allclose(T[:3, :3], cs.R_enu_body(0, 0, 0))


def test_vehicle_aggregator_releases_pings_when_navigation_covers_them():
    ag = vb.VehicleAggregator()
    ag.push_ping((5.0, "ping"))
    assert ag.pop_ready() == []
    ag.push_pose(0.0, cs.pose_enu([0, 0, -10], 0, 0, 0))
    ag.push_pose(10.0, cs.pose_enu([100, 0, -10], 0, 0, 0))
    assert not ag.push_pose(9.0, np.eye(4))                                   # out of order rejected
    out = ag.pop_ready()
    assert len(out) == 1 and np.allclose(out[0][1][:3, 3], [50, 0, -10])
