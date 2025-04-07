import numpy as np


# Approximates color temperature from xy coordinates.
# Error in range Illuminant A to D65 is < 3 Kelvins.
# n = (x - 0.3320) / (0.1858 - y);
# CCT = 437*n^3 + 3601*n^2 + 6861*n + 5517
def temp_from_xy(xy):
    n = (xy[0] - 0.3320) / (0.1858 - xy[1])
    return 437 * n * n * n + 3601 * n * n + 6861 * n + 5517


# Computes a 3x3 matrix which maps colors from white point `from_white` to
# white point `to_white` using the linearized Bradford adaptation method.
def map_colors_between_white_points(from_white,
                                    to_white,
                                    eps=1e-12,
                                    min_scale=0.1,
                                    max_scale=10.0):
    w1 = linearized_bradford_matrix() @ xyz_from_xy(from_white)
    w2 = linearized_bradford_matrix() @ xyz_from_xy(to_white)

    # Negative white coordinates are kind of meaningless (clip to 0).
    w1 = np.clip(w1, 0, None)
    w2 = np.clip(w2, 0, None)

    # Calculate ratio.
    a = w2 / (w1 + eps)

    # Limit scaling to something reasonable.
    a = np.clip(a, min_scale, max_scale)

    return np.linalg.inv(linearized_bradford_matrix()) @ np.diag(
        a) @ linearized_bradford_matrix()


# Converts xy (chromaticity) to XYZ (tristimulus).
# 1. First clamps xy to the range of real (x, y) coordinates.
# 2. Then returns (x / y, 1, (1 - x - y) / y).
def xyz_from_xy(xy, min_val=1e-6, max_val=(1.0 - 1e-6)):
    temp = xy

    # Restrict xy coord to somewhere in the range of real xy coordinates.
    # This prevents math from doing strange things when users specify extreme
    # temperature/tint coordinates.
    temp = temp.clip(min_val, max_val)

    if temp.sum() > max_val:
        scale = max_val / temp.sum()
        temp *= scale

    return np.array([temp[0] / temp[1], 1, (1 - temp[0] - temp[1]) / temp[1]])


# Converts XYZ (tristimulus) to xy (chromaticity).
# - Returns (x, y) / (x + y + z).
# - If the sum of the input vector is 0, returns `d50_XYCoord()`.
def xy_from_xyz(xyz):
    if xyz.sum() > 0:
        return xyz[:2] / xyz.sum()
    else:
        # Return the PCS white if the sum is <= 0.0.
        return d50_XYCoord()


# Standard matrix yielding linear sRGB from XYZ D50 (aka PCS).
def linear_srgb_from_xyz_d50():
    # xyz_d65 <- xyz_d50.
    xyz_d65_from_xyz_d50 = map_colors_between_white_points(d50_xy(), d65_xy())

    # sRGB <- xyz_d65.
    return srgb_from_xyz_d65() @ xyz_d65_from_xyz_d50


# Standard matrix yielding linear sRGB from XYZ D65.
def srgb_from_xyz_d65():
    return np.array([[3.24071, -1.53726, -0.498571],
                     [-0.969258, 1.87599, 0.0415557],
                     [0.0556352, -0.203996, 1.05707]])


# Standard linearized Bradford matrix.
# See http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html.
def linearized_bradford_matrix():
    return np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])


# Some standard white points expressed as (x, y) coordinates.
def d50_xy():
    return np.array([0.3457, 0.3585])


def d55_xy():
    return np.array([0.3127, 0.3290])


def d65_xy():
    return np.array([0.3127, 0.3290])


def d75_xy():
    return np.array([0.2990, 0.3149])


def standard_a_xy():
    return np.array([0.4476, 0.4074])
