import numpy as np
from color_info import ColorInfo
from color_utils import d50_xy, temp_from_xy, xy_from_xyz, xyz_from_xy, map_colors_between_white_points, linear_srgb_from_xyz_d50


class ColorSpec:

    def __init__(self, color_info: ColorInfo):
        self.color_info = color_info.standardized()

    # Returns the camera neutral which describes the three R, G, B values in
    # camera color space that are considered "white" for the given white point.
    # The returned camera neutral values are normalized such that the largest
    # value is 1.
    def camera_neutral_for(self, white_xy: np.ndarray) -> np.ndarray:
        camera_neutral = self.camera_from_xyz(white_xy) @ xyz_from_xy(white_xy)

        # Normalize such that the largest value is 1.0f.
        return camera_neutral / camera_neutral.max()

    # Given white point `white_xy` as an (x, y) chromaticity, returns the
    # transform that yields ABC (camera coordinates) from XYZ (with the given
    # white point).
    def camera_from_xyz(self, white_xy: np.ndarray) -> np.ndarray:
        # Convert to temperature/offset space.
        white_temp = temp_from_xy(white_xy)
        temp1 = self.color_info.calibration_illuminant_1_temperature()
        temp2 = self.color_info.calibration_illuminant_2_temperature()

        # Find g, the fraction to weight the first calibration.
        if white_temp <= temp1:
            g = 1.0
        elif white_temp >= temp2:
            g = 0.0
        else:
            inverse_temp = 1.0 / white_temp
            g = (inverse_temp - (1.0 / temp2)) / ((1.0 / temp1) -
                                                  (1.0 / temp2))

        # Interpolate the color matrix using g.
        if g >= 1.0:
            color_matrix = self.color_info.calibrated_color_matrix_1()
        elif g <= 0.0:
            color_matrix = self.color_info.calibrated_color_matrix_2()
        else:
            color_matrix = g * self.color_info.calibrated_color_matrix_1() + (
                1.0 - g) * self.color_info.calibrated_color_matrix_2()

        return color_matrix

    #  Given a neutral in ABC (camera) coordinates, computes the (x, y)
    # chromaticity that would map to that neutral using an iterative algorithm.
    #
    #  The components of `camera_neutral` range from 0.0 to 1.0 and should be
    #  normalized such that the largest value is 1.0 .
    def xy_from_camera_neutral(self,
                               camera_neutral: np.ndarray,
                               max_passes: int = 30,
                               eps: float = 1e-6) -> np.ndarray:
        # Start with D50 as an initial guess.
        last = d50_xy()

        for p in range(max_passes):
            camera_from_xyz = self.camera_from_xyz(last)

            # Linear solve to guess a neutral in XYZ space with the current
            # ABC <- XYZ transform.
            neutral_xyz = np.linalg.solve(camera_from_xyz, camera_neutral)

            # Get its xy chromaticity.
            next = xy_from_xyz(neutral_xyz)

            if np.abs(next - last).sum() < eps:
                return next

            # If we reach the limit without converging, we are most likely in a two
            # value oscillation. So take the average of the last two estimates and give
            # up.
            if p == max_passes - 1:
                print("Failed to converge to an xy white point from the given "
                      "camera neutral. Returning the average of the last two "
                      "iterations.")
                next = 0.5 * (last + next)

            last = next

        return last

    # Returns the matrix mapping ABC (camera) coordinates to PCS (XYZ D50)
    # coordinates given the white point in (x, y) space.
    def pcs_from_xy(self, white_xy: np.ndarray) -> np.ndarray:
        # Find transformation camera <- XYZ(white_xy).
        camera_from_xyz = self.camera_from_xyz(white_xy)

        # Find the transformation XYZ(white_xy) <- XYZ(D50).
        xyz_white_xy_from_xyz_d50 = map_colors_between_white_points(
            d50_xy(), white_xy)

        # Combine to get camera <- XYZ(D50).
        camera_from_pcs = camera_from_xyz @ xyz_white_xy_from_xyz_d50

        # Find `d50_camera`, a point in camera space that corresponds to D50.
        # Then scale `camera_from_pcs` its max element.
        d50_xyz = xyz_from_xy(d50_xy())
        d50_camera = camera_from_pcs @ d50_xyz
        max_coeff = d50_camera.max()
        assert max_coeff > 0.0

        inv_scale = 1.0 / max_coeff

        scaled_camera_from_pcs = inv_scale * camera_from_pcs

        # TODO(jiawen): Handle the case where we have forward matrices.
        # Don't forget to scale the white.
        return np.linalg.inv(scaled_camera_from_pcs)

    # Returns the matrix mapping ABC (camera) coordinates to PCS (XYZ D50)
    # coordinates given a camera neutral.
    def pcs_from_camera_neutral(self,
                                camera_neutral: np.ndarray) -> np.ndarray:
        white_xy = self.xy_from_camera_neutral(camera_neutral)
        return self.pcs_from_xy(white_xy)

    # Returns linear_srgb_from_xyz_d50() @ pcs_from_xy(white_xy).
    def linear_srgb_from_xy(self, white_xy: np.ndarray) -> np.ndarray:
        pcs_from_camera = self.pcs_from_xy(white_xy)
        return linear_srgb_from_xyz_d50() @ pcs_from_camera

    # Returns linear_srgb_from_xyz_d50() @ pcs_from_camera_neutral(camera_neutral).
    def linear_srgb_from_camera(self,
                                camera_neutral: np.ndarray) -> np.ndarray:
        pcs_from_camera = self.pcs_from_camera_neutral(camera_neutral)
        return linear_srgb_from_xyz_d50() @ pcs_from_camera
