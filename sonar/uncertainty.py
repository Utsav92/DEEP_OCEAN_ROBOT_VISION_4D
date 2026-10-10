"""First-order measurement uncertainty propagation for multibeam soundings.

For a return at slant range r and across-track angle theta (planar model: x = r sin(theta), z = r cos(theta)):
  var_x = (sin(theta) s_r)^2 + (r cos(theta) s_theta)^2
  var_z = (cos(theta) s_r)^2 + (r sin(theta) s_theta)^2
Navigation position uncertainty (horizontal, vertical) and attitude-induced lever-arm error add in quadrature.
"""
import numpy as np


def sounding_sigma(r, theta, sigma_r, sigma_theta, sigma_nav_xy=0.0, sigma_nav_z=0.0, sigma_att=0.0):
    r, theta = np.asarray(r), np.asarray(theta)
    sx2 = (np.sin(theta) * sigma_r) ** 2 + (r * np.cos(theta) * sigma_theta) ** 2
    sz2 = (np.cos(theta) * sigma_r) ** 2 + (r * np.sin(theta) * sigma_theta) ** 2
    att2 = (r * sigma_att) ** 2                        # attitude error swings the whole ray
    sh = np.sqrt(sx2 + sigma_nav_xy ** 2 + att2 * 0.5)
    sv = np.sqrt(sz2 + sigma_nav_z ** 2 + att2 * 0.5)
    return sh, sv


def range_sigma(r, frac, floor_m):
    return floor_m + frac * np.asarray(r)
