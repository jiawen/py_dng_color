from dataclasses import dataclass
import numpy as np
from color_utils import d65_xy, standard_a_xy, temp_from_xy


@dataclass(frozen=True)
class ColorInfo:
    # Color correction matrices.
    color_matrix_1: np.ndarray = np.eye(3)
    color_matrix_2: np.ndarray = np.eye(3)

    # Correlated color temperature of the calibration illuminants.
    # TODO(jiawen): Implement full DNG light source spec.
    calibration_illuminant_1_xy: np.ndarray = d65_xy()
    calibration_illuminant_2_xy: np.ndarray = standard_a_xy()

    # Calibration matrices to adjust color matrices between the particular unit
    # and color_matrix.
    camera_calib_1: np.ndarray = np.eye(3)
    camera_calib_2: np.ndarray = np.eye(3)

    # Analog color balance: 3x1 vector.
    analog_balance: np.ndarray = np.ones(3)

    def calibration_illuminant_1_temperature(self) -> float:
        return temp_from_xy(self.calibration_illuminant_1_xy)

    def calibration_illuminant_2_temperature(self) -> float:
        return temp_from_xy(self.calibration_illuminant_2_xy)

    def calibrated_color_matrix_1(self) -> np.ndarray:
        return np.diag(
            self.analog_balance) @ self.camera_calib_1 @ self.color_matrix_1

    def calibrated_color_matrix_2(self) -> np.ndarray:
        return np.diag(
            self.analog_balance) @ self.camera_calib_2 @ self.color_matrix_2

    # Returns a "standardized" ColorInfo where temp1 <= temp2.
    # TODO(jiawen): Python 3.11. -> Self.
    def standardized(self):
        if self.calibration_illuminant_1_temperature(
        ) <= self.calibration_illuminant_2_temperature():
            return self
        else:
            return ColorInfo(
                color_matrix_1=self.color_matrix_2,
                color_matrix_2=self.color_matrix_1,
                calibration_illuminant_1_xy=self.calibration_illuminant_2_xy,
                calibration_illuminant_2_xy=self.calibration_illuminant_1_xy,
                camera_calib_1=self.camera_calib_2,
                camera_calib_2=self.camera_calib_1,
                analog_balance=self.analog_balance,
            )
