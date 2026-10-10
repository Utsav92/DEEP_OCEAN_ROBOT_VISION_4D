"""Gridded bathymetry loading (GeoTIFF from NOAA NCEI, NetCDF in GEBCO layout) and metric resampling.

Elevation convention: metres, positive UP, negative below the vertical datum (GEBCO: MSL-ish; NCEI CUDEM: NAVD88).
Depth = -elevation. NoData becomes NaN and is never filled with invented values.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from .coordinate_systems import LocalFrame


@dataclass
class GeoGrid:
    lon: np.ndarray          # (nx,) ascending, degrees
    lat: np.ndarray          # (ny,) ascending, degrees
    z: np.ndarray            # (ny, nx) elevation m (up), NaN = no data
    source: str = ""
    vertical_datum: str = "unknown"
    crs: str = "EPSG:4326"
    meta: dict = field(default_factory=dict)

    @property
    def bbox(self):
        return (float(self.lon[0]), float(self.lat[0]), float(self.lon[-1]), float(self.lat[-1]))

    def valid_fraction(self):
        return float(np.isfinite(self.z).mean())


@dataclass
class MetricGrid:
    """Regular grid in the local ENU frame. z[iy, ix] at (x[ix], y[iy]); NaN = not observed / no data."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    frame: LocalFrame
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])

    @property
    def shape(self):
        return self.z.shape

    def sample(self, X, Y, fill=np.nan):
        """Bilinear elevation at (X, Y); NaN where the stencil touches NaN or leaves the grid."""
        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        cx = (X - self.x[0]) / self.dx
        cy = (Y - self.y[0]) / self.dy
        z = np.where(np.isfinite(self.z), self.z, np.nan)
        out = _bilinear_nan(z, cy, cx)
        return np.where(np.isfinite(out), out, fill)

    def gradient(self):
        gy, gx = np.gradient(np.where(np.isfinite(self.z), self.z, np.nan), self.dy, self.dx)
        return gx, gy

    def slope_deg(self):
        gx, gy = self.gradient()
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def normals(self):
        gx, gy = self.gradient()
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _bilinear_nan(z, cy, cx):
    """Bilinear interpolation that propagates NaN honestly (any NaN corner -> NaN)."""
    ny, nx = z.shape
    inside = (cx >= 0) & (cx <= nx - 1) & (cy >= 0) & (cy <= ny - 1)
    x0 = np.clip(np.floor(cx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(cy).astype(int), 0, ny - 2)
    fx, fy = cx - x0, cy - y0
    v = ((1 - fx) * (1 - fy) * z[y0, x0] + fx * (1 - fy) * z[y0, x0 + 1]
         + (1 - fx) * fy * z[y0 + 1, x0] + fx * fy * z[y0 + 1, x0 + 1])
    return np.where(inside, v, np.nan)


def clean_elevation(z, nodata_values=(-99999.0, -32768.0, -9999.0), max_abs=12000.0, nodata=None):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z) | (np.abs(z) > max_abs)
    for v in list(nodata_values) + ([nodata] if nodata is not None else []):
        bad |= z == v
    z[bad] = np.nan
    return z


def load_geotiff(path, nodata_values=(-99999.0, -32768.0, -9999.0), max_abs=12000.0, vertical_datum="unknown"):
    import rasterio
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError(f"{path}: expected geographic WGS84 (EPSG:4326), got {ds.crs}")
        a = ds.read(1).astype(np.float64)
        T = ds.transform
        h, w = a.shape
        lon = T.c + (np.arange(w) + 0.5) * T.a
        lat = T.f + (np.arange(h) + 0.5) * T.e          # T.e < 0 (north-up raster)
        z = clean_elevation(a, nodata_values, max_abs, ds.nodata)
    if lat[0] > lat[-1]:                                 # make lat ascending
        lat, z = lat[::-1], z[::-1]
    return GeoGrid(lon=lon, lat=lat, z=z, source=str(path), vertical_datum=vertical_datum,
                   meta={"cell_deg": (float(abs(T.a)), float(abs(T.e)))})


def load_netcdf(path, var=None, lon_name=None, lat_name=None, vertical_datum="GEBCO (MSL)"):
    """GEBCO-style NetCDF: 1-D `lon`/`lat` coordinate variables and a 2-D `elevation` (m, up positive) variable."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        names = list(ds.variables)
        lon_name = lon_name or next(n for n in names if n.lower() in ("lon", "longitude", "x"))
        lat_name = lat_name or next(n for n in names if n.lower() in ("lat", "latitude", "y"))
        var = var or next(n for n in names if n.lower() in ("elevation", "z", "depth", "band1") and ds[n].ndim == 2)
        lon = np.array(ds[lon_name][:], dtype=np.float64)
        lat = np.array(ds[lat_name][:], dtype=np.float64)
        v = ds[var]
        z = np.ma.filled(v[:].astype(np.float64), np.nan)
        units = getattr(v, "units", "m")
        positive = str(getattr(v, "positive", "up")).lower()
    if var.lower() == "depth" or positive == "down":
        z = -z                                           # convert depth-positive-down to elevation up
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    return GeoGrid(lon=lon, lat=lat, z=clean_elevation(z), source=str(path), vertical_datum=vertical_datum,
                   meta={"variable": var, "units": units})


def crop(grid, bbox):
    lo0, la0, lo1, la1 = bbox
    ix = np.where((grid.lon >= lo0) & (grid.lon <= lo1))[0]
    iy = np.where((grid.lat >= la0) & (grid.lat <= la1))[0]
    if ix.size < 2 or iy.size < 2:
        raise ValueError("crop box does not overlap the grid")
    return GeoGrid(grid.lon[ix], grid.lat[iy], grid.z[np.ix_(iy, ix)], grid.source, grid.vertical_datum,
                   grid.crs, dict(grid.meta))


def to_metric_grid(grid, frame=None, spacing=None, margin_cells=0):
    """Resample a geographic grid onto a regular metric ENU grid (bilinear, NaN-propagating).

    Latitude/longitude degrees are NEVER treated as metres: every output node is projected back to lon/lat
    through the local transverse-Mercator frame and the source grid is sampled there.
    """
    if frame is None:
        frame = LocalFrame(0.5 * (grid.lon[0] + grid.lon[-1]), 0.5 * (grid.lat[0] + grid.lat[-1]))
    # extent of the source grid in local metres (corners + edge midpoints)
    lons = np.array([grid.lon[0], grid.lon[-1], grid.lon[0], grid.lon[-1], grid.lon.mean(), grid.lon.mean()])
    lats = np.array([grid.lat[0], grid.lat[0], grid.lat[-1], grid.lat[-1], grid.lat[0], grid.lat[-1]])
    ex, ey = frame.to_local(lons, lats)
    if spacing is None:
        cell_m = min(abs(np.diff(grid.lon).mean()) * 111320.0 * np.cos(np.radians(frame.lat0)),
                     abs(np.diff(grid.lat).mean()) * 110574.0)
        spacing = float(cell_m)
    x0, x1 = ex.min() + margin_cells * spacing, ex.max() - margin_cells * spacing
    y0, y1 = ey.min() + margin_cells * spacing, ey.max() - margin_cells * spacing
    # shrink to the inscribed rectangle so nodes stay inside the source grid (TM grid is slightly rotated vs lon/lat)
    x0, x1 = np.ceil(max(ex[0], ex[2]) / spacing) * spacing, np.floor(min(ex[1], ex[3]) / spacing) * spacing
    y0, y1 = np.ceil(max(ey[0], ey[1]) / spacing) * spacing, np.floor(min(ey[2], ey[3]) / spacing) * spacing
    xs = np.arange(x0, x1 + 0.5 * spacing, spacing)
    ys = np.arange(y0, y1 + 0.5 * spacing, spacing)
    X, Y = np.meshgrid(xs, ys)
    lon, lat = frame.to_geo(X, Y)
    cx = (lon - grid.lon[0]) / (grid.lon[-1] - grid.lon[0]) * (len(grid.lon) - 1)
    cy = (lat - grid.lat[0]) / (grid.lat[-1] - grid.lat[0]) * (len(grid.lat) - 1)
    z = _bilinear_nan(grid.z, cy, cx)
    return MetricGrid(xs, ys, z, frame, grid.source,
                      {"spacing_m": float(spacing), "vertical_datum": grid.vertical_datum,
                       "frame": frame.meta(), "valid_fraction": float(np.isfinite(z).mean())})


def fill_for_raycast(z):
    """Fill NaN with the nearest valid value ONLY for numerical ray marching in the simulator (never for maps)."""
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(z)
    if not bad.any():
        return z
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return z[tuple(idx)]
