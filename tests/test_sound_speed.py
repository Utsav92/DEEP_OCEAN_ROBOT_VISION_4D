import numpy as np
import pytest

from preprocessing import sound_speed as ss


def test_mackenzie_reference_values():
    # standard check value: T=25 C, S=35 psu, D=1000 m  -> 1550.74 m/s (Mackenzie 1981 / NPL table)
    assert ss.mackenzie(25.0, 35.0, 1000.0) == pytest.approx(1550.74, abs=0.1)
    assert ss.mackenzie(10.0, 35.0, 0.0) == pytest.approx(1489.8, abs=0.5)


def test_twtt_range_roundtrip():
    r = np.array([10.0, 123.4, 250.0])
    c = 1500.0
    assert np.allclose(ss.range_from_twtt(ss.twtt_from_range(r, c), c), r)
    assert ss.range_from_twtt(0.2, 1500.0) == pytest.approx(150.0)      # r = c * dt / 2


def test_harmonic_mean_constant_profile():
    p = ss.SoundSpeedProfile([0, 2000], [1500.0, 1500.0])
    assert p.harmonic_mean(100.0, 900.0) == pytest.approx(1500.0)


def test_harmonic_mean_between_endpoint_speeds():
    p = ss.SoundSpeedProfile.from_tsd([[0, 14, 33.4], [1000, 4, 34.4]])
    m = float(p.harmonic_mean(0.0, 1000.0))
    assert min(p.at(0.0), p.at(1000.0)) - 5 < m < max(p.at(0.0), p.at(1000.0)) + 5


def test_profile_range_matches_truth_conversion():
    p = ss.SoundSpeedProfile.from_tsd([[0, 14, 33.4], [300, 7, 34.1], [1000, 4, 34.4]])
    depth_sonar, theta, r = 600.0, np.radians(35.0), 90.0
    c = float(p.harmonic_mean(depth_sonar, depth_sonar + r * np.cos(theta)))
    twtt = 2 * r / c
    r_back = ss.range_from_twtt_profile(np.array([twtt]), np.array([theta]), np.array([depth_sonar]), p)
    assert r_back[0] == pytest.approx(r, abs=0.02)


def test_ray_trace_reduces_to_straight_ray_in_constant_profile():
    p = ss.SoundSpeedProfile([0, 5000], [1500.0, 1500.0])
    theta, r = np.radians(40.0), 100.0
    x, z, chord = ss.ray_trace(np.array(theta), np.array(2 * r / 1500.0), p, 100.0)
    assert x == pytest.approx(r * np.sin(theta), rel=1e-3)
    assert z == pytest.approx(r * np.cos(theta), rel=1e-3)


def test_calibrated_ranges_are_not_converted_again():
    from preprocessing.sonar_loader import Pings
    from sonar.multibeam import ranges_from_pings
    n, b = 2, 3
    pings = Pings(t=np.arange(n, dtype=float), twtt=np.full((n, b), 0.2), beam_theta=np.zeros(b),
                  beam_phi=np.zeros(b), intensity=np.ones((n, b)), quality=np.full((n, b), 3),
                  c_sonar=np.full(n, 1500.0), T_vehicle_sonar=np.eye(4), ranges_m=np.full((n, b), 42.0))
    r = ranges_from_pings(pings, None, np.zeros(n))
    assert np.all(r == 42.0)
