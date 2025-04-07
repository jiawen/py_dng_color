import numpy as np
from color_info import ColorInfo
from color_spec import ColorSpec

COLOR_MATRIX_1 = np.array([[0.7858, -0.2151, -0.091], [-0.5955, 1.431, 0.1737],
                           [-0.2399, 0.3391, 0.579]])
COLOR_MATRIX_2 = np.array([[1.0753, -0.3149,
                            -0.2765], [-0.553, 1.6283, -0.1152],
                           [-0.0538, 0.192, 0.5914]])
CALIBRATION_ILLUMINANT_1_XY = np.array([0.3127, 0.3290])
CALIBRATION_ILLUMINANT_2_XY = np.array([0.4476, 0.4074])

COLOR_INFO = ColorInfo(
    color_matrix_1=COLOR_MATRIX_1,
    color_matrix_2=COLOR_MATRIX_2,
    calibration_illuminant_1_xy=CALIBRATION_ILLUMINANT_1_XY,
    calibration_illuminant_2_xy=CALIBRATION_ILLUMINANT_2_XY,
)

# The "as shot neutral": what the camera decided the neutral should be (aka the vendor's white
# balance).
AS_SHOT_NEUTRAL = np.array([0.472441, 1.0, 0.629921])

# Make a ColorSpec to conveniently access quantities derived from ColorInfo.
color_spec = ColorSpec(COLOR_INFO)

linear_srgb_from_camera = color_spec.linear_srgb_from_camera(AS_SHOT_NEUTRAL)

print(linear_srgb_from_camera)

# Should be:
# [[ 3.33052497 -0.57955022  0.00951405]
# [-0.3148691   1.31931082 -0.27073903]
# [ 0.22634638 -0.62227944  2.40575759]]