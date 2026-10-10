import numpy as np
import pytest

from mapping import bathymetry_grid as bgrid, occupancy_map as om, obstacle_candidates as oc, pointcloud_fusion as pf
from mapping import depth_contours as dc, map_export as me, uncertainty_map as um
from simulation.terrain import SimTerrain
from preprocessing.bathymetry_loader import MetricGrid
from preprocessing.coordinate_systems import LocalFrame


def _points(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
    z = -500 + 0.1 * x + 3 * np.sin(y / 20)
    return np.column_stack([x, y, z + rng.normal(0, 0.1, n)])


def test_grid_mean_and_coverage():
    P = _points()
    g = bgrid.grid_soundings(P, np.arange(len(P), dtype=float), np.full(len(P), 0.5), np.full(len(P), 0.2),
                             np.full(len(P), 0.3), 4.0, (0, 200, 0, 200))
    truth = -500 + 0.1 * (g.x[None, :]) + 3 * np.sin(g.y[:, None] / 20)
    m = g.observed
    assert m.mean() > 0.95
    assert np.nanmean(np.abs(g.z[m] - truth[m])) < 1.0         # cell mean of a surface sloping 0.4 m/cell
    assert np.all(g.t_first[m] <= g.t_last[m])
    assert np.isnan(g.z[~m]).all()                              # unobserved cells never get a value


def test_interpolation_flagged_and_limited():
    P = _points(3000, 1)
    g = bgrid.grid_soundings(P, np.zeros(len(P)), np.zeros(len(P)), np.ones(len(P)), np.ones(len(P)), 2.0,
                             (0, 200, 0, 200), max_gap_cells=2)
    assert g.interpolated.sum() > 0
    assert not np.any(g.interpolated & g.observed)              # filled cells are never mistaken for observed ones
    far = ~g.observed & ~g.interpolated
    assert np.isnan(g.z_interp[far]).all()


def test_occupancy_free_occupied_unknown():
    occ = om.OccupancyMap((0, 100, 0, 100), (-100, 0), voxel=10.0)
    occ.integrate(np.array([[50.0, 50.0, -10.0]] * 4), np.array([[50.0, 50.0, -90.0]] * 4))
    c = occ.classify()
    s = occ.stats()
    assert (c == 2).sum() >= 1 and (c == 1).sum() >= 5
    assert s["unknown_fraction"] > 0.9                          # untouched water is UNKNOWN, not free
    iz, iy, ix, ok = occ._idx(np.array([[50.0, 50.0, -50.0]]))
    assert c[iz[0], iy[0], ix[0]] == 1
    iz, iy, ix, ok = occ._idx(np.array([[5.0, 5.0, -50.0]]))
    assert c[iz[0], iy[0], ix[0]] == 0


def test_obstacle_detection_finds_bump_not_slope():
    x = y = np.arange(0, 400, 4.0)
    X, Y = np.meshgrid(x, y)
    z = -800 + 0.15 * X                                          # gentle slope
    z = z + 9.0 * ((np.abs(X - 200) < 8) & (np.abs(Y - 200) < 6))   # 16 x 12 m, 9 m tall box
    g = bgrid.BathyGrid(0.0, 0.0, 4.0, z, np.full(z.shape, 20, np.int32), np.full(z.shape, 0.3), np.zeros_like(z),
                        np.ones_like(z), np.full(z.shape, 0.5), z.copy(), np.zeros(z.shape, bool))
    c, top = oc.detect(g, 36.0, 3.5, 60.0)
    assert len(c) == 1
    p = c[0]["position_enu_m"]
    assert abs(p[0] - 200) < 8 and abs(p[1] - 200) < 8
    assert c[0]["classification_status"] == "unclassified"       # never an invented semantic label
    ev = oc.match_to_truth(c, [{"x": 200.0, "y": 200.0}])
    assert ev["recall"] == 1.0


def test_voxel_downsample_and_display_subset_stability():
    P = np.random.default_rng(0).uniform(0, 10, (5000, 3))
    c, cnt, first = pf.voxel_downsample(P, 2.0)
    assert cnt.sum() == len(P) and len(c) <= 6 ** 3
    a, b = pf.select_display_subset(100000, 1000), pf.select_display_subset(100000, 1000)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0)


def test_contours_levels():
    x = y = np.arange(0, 100.0)
    X, Y = np.meshgrid(x, y)
    z = -500 + 2.0 * X
    cs_ = dc.contours(x, y, z, interval=50.0)
    levels = sorted({c["level"] for c in cs_})
    assert levels and all(abs(l / 50 - round(l / 50)) < 1e-9 for l in levels)
    seg = dc.contour_segments_enu(cs_)
    assert seg.shape[1:] == (2, 3)


def test_texture_resample_preserves_nan():
    x = y = np.arange(0, 100.0)
    z = np.add.outer(y, x) * 0.1
    z[40:60, 40:60] = np.nan
    tex, ok, ext = me.resample_to_texture(x, y, z, 50)
    assert np.isnan(tex[ok == 0]).all() and (~ok).sum() > 0
    assert np.isfinite(tex[0, 0])


def test_uncertainty_display_bounded_and_nan_kept():
    s = np.array([0.1, 1.0, 5.0, np.nan])
    d = um.display_value(s)
    assert d[0] == 0 and d[2] == 1 and np.isnan(d[3])


def test_simulator_raycast_hits_plane_at_known_range():
    x = y = np.arange(-500, 501, 5.0)
    g = MetricGrid(x, y, np.full((len(y), len(x)), -100.0), LocalFrame(0, 0))
    t = SimTerrain(g)
    r, hit, n = t.cast(np.array([[0.0, 0.0, -20.0]]), np.array([[0.0, 0.0, -1.0]]), 200.0)
    assert r[0] == pytest.approx(80.0, abs=0.01)
    r2, hit2, _ = t.cast(np.array([[0.0, 0.0, -20.0]]), np.array([[0.6, 0.0, -0.8]]), 200.0)
    assert r2[0] == pytest.approx(100.0, abs=0.05)               # 80 m down along a 0.8 z-component
    r3, _, _ = t.cast(np.array([[0.0, 0.0, -20.0]]), np.array([[0.0, 0.0, 1.0]]), 200.0)
    assert np.isnan(r3[0])                                       # looking up never returns
