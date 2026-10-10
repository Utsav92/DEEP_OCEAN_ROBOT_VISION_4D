"""Forward-looking sonar (FLS) imagery: native range-bearing images, intensity normalisation, noise filtering, Cartesian
reprojection with a valid mask.

IMPORTANT: an FLS pixel is an intensity at (range, bearing); the elevation angle inside the vertical aperture is
unknown. Intensity is therefore NOT converted to metric depth anywhere in this module.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from .intensity_processing import normalise, speckle_lee
from .range_bearing import cartesian_grid, cart_to_polar


def prepare(image_u8, gain=1.0):
    img = np.asarray(image_u8, dtype=np.float64) / 255.0
    img = speckle_lee(img, 3, enl=6.0)
    img = normalise(img, percentile=(1.0, 99.5))
    return np.clip(img * gain, 0.0, 1.0)


def valid_range_mask(shape, max_range, min_range=1.0):
    nr, nb = shape
    r = (np.arange(nr) + 0.5) / nr * max_range
    return np.repeat((r >= min_range)[:, None], nb, 1)


def to_cartesian(img, max_range, bearings, size=256):
    """Reproject a (n_range, n_bearing) polar image to a Cartesian image (rows = forward, cols = starboard).
    Returns (cart, valid_mask, X, Y). Pixels outside the fan are invalid (NaN-free, mask False)."""
    nr, nb = img.shape
    fov = bearings[-1] - bearings[0]
    X, Y = cartesian_grid(max_range, fov, size)
    r, b = cart_to_polar(X, Y)
    ri = r / max_range * nr - 0.5
    bi = (b - bearings[0]) / (bearings[-1] - bearings[0]) * (nb - 1)
    inside = (r <= max_range) & (b >= bearings[0]) & (b <= bearings[-1])
    cart = map_coordinates(img, [ri, bi], order=1, mode="constant", cval=0.0)
    return np.where(inside, cart, 0.0), inside, X, Y
