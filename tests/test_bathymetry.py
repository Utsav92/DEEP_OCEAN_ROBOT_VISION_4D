import os

import numpy as np
import pytest

from preprocessing import bathymetry_loader as bl
from preprocessing.coordinate_systems import LocalFrame
from mapping import terrain_mesh as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _synthetic_geo(nodata=None):
    lon = np.linspace(-122.0, -121.99, 41)
    lat = np.linspace(36.77, 36.78, 41)
    X, Y = np.meshgrid(lon, lat)
    z = -800.0 + 3000.0 * (X + 122.0) + 2000.0 * (Y - 36.77)       # a plane: exact bilinear
    if nodata is not None:
        z[10:13, 10:13] = nodata
    return bl.GeoGrid(lon, lat, bl.clean_elevation(z), "synthetic", "test")


def test_nodata_becomes_nan_never_filled():
    g = _synthetic_geo(nodata=-99999.0)
    assert np.isnan(g.z[10:13, 10:13]).all()
    m = bl.to_metric_grid(g, spacing=20.0)
    assert np.isnan(m.z).any()                      # holes survive resampling


def test_metric_resample_preserves_plane():
    g = _synthetic_geo()
    fr = LocalFrame(g.lon.mean(), g.lat.mean())
    m = bl.to_metric_grid(g, fr, spacing=25.0)
    lon, lat = fr.to_geo(*np.meshgrid(m.x, m.y))
    truth = -800.0 + 3000.0 * (lon + 122.0) + 2000.0 * (lat - 36.77)
    assert np.nanmax(np.abs(m.z - truth)) < 1e-6
    assert m.dx == pytest.approx(25.0)


def test_metric_extent_is_in_metres_not_degrees():
    g = _synthetic_geo()
    m = bl.to_metric_grid(g, spacing=20.0)
    width = m.x[-1] - m.x[0]
    assert 700 < width < 1000                       # 0.01 deg of longitude at 36.8 N ~ 890 m


def test_elevation_convention_depth_conversion(tmp_path):
    import netCDF4
    p = tmp_path / "gebco_like.nc"
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("lat", 5)
        ds.createDimension("lon", 6)
        la = ds.createVariable("lat", "f8", ("lat",))
        lo = ds.createVariable("lon", "f8", ("lon",))
        el = ds.createVariable("elevation", "i2", ("lat", "lon"))
        la[:] = np.linspace(36.0, 36.04, 5)[::-1]          # descending latitude on purpose
        lo[:] = np.linspace(-122.0, -121.95, 6)
        el[:] = np.full((5, 6), -1234, dtype=np.int16)
    g = bl.load_netcdf(str(p))
    assert g.lat[0] < g.lat[-1]
    assert np.allclose(g.z, -1234.0)
    assert g.vertical_datum.startswith("GEBCO")


def test_triangulation_skips_nodata_cells():
    valid = np.ones((4, 4), bool)
    assert len(tm.grid_triangles(valid)) == 2 * 3 * 3
    valid[1, 1] = False                              # removes the 4 cells touching that vertex
    assert len(tm.grid_triangles(valid)) == 2 * (9 - 4)


def test_mesh_normals_point_up_on_flat_terrain():
    x, y = np.arange(0, 50, 5.0), np.arange(0, 50, 5.0)
    g = bl.MetricGrid(x, y, np.full((10, 10), -500.0), LocalFrame(0, 0))
    mesh = tm.terrain_mesh(g)
    assert np.allclose(mesh["normals"][:, 2], 1.0, atol=1e-6)


def test_obj_export_roundtrip(tmp_path):
    x, y = np.arange(0, 40, 5.0), np.arange(0, 40, 5.0)
    z = np.add.outer(y, x) * 0.1 - 500
    z[2, 2] = np.nan
    g = bl.MetricGrid(x, y, z, LocalFrame(0, 0))
    mesh = tm.terrain_mesh(g)
    meta = tm.export_mesh_bundle(mesh, str(tmp_path), "t", {"k": 1})
    V, T = tm.read_obj(str(tmp_path / "t.obj"))
    assert len(T) == meta["triangles"] == len(mesh["triangles"])
    assert V[:, 2].min() >= z[np.isfinite(z)].min() - 1e-3


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "bathymetry", "monterey_survey_native.tif")),
                    reason="real NOAA tile not downloaded")
def test_real_noaa_tile_loads_and_is_plausible():
    g = bl.load_geotiff(os.path.join(ROOT, "data", "bathymetry", "monterey_survey_native.tif"))
    assert g.valid_fraction() > 0.99
    assert -1200 < np.nanmin(g.z) < np.nanmax(g.z) < -400          # Monterey Canyon at ~565-1030 m depth
    m = bl.to_metric_grid(g, spacing=5.0)
    assert 2000 < (m.x[-1] - m.x[0]) < 2600
