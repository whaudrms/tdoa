"""Linearized 2D TDOA localization using a pseudoinverse."""

import numpy as np


SOUND_SPEED = 343.0  # m/s


def estimate_position(sensors, tdoa, c=SOUND_SPEED):
    """Solve A @ [x, y, r1] = b using measured differences relative to S1."""
    sensors = np.asarray(sensors, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float)
    if sensors.ndim != 2 or sensors.shape[1] != 2 or len(sensors) < 4:
        raise ValueError('At least four 2D sensor coordinates are required')
    if tdoa.shape != (len(sensors) - 1,):
        raise ValueError('Provide one TDOA value per non-reference sensor')
    if not np.isfinite(sensors).all() or not np.isfinite(tdoa).all():
        raise ValueError('Sensor coordinates and TDOA must be finite')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('Sound speed must be positive and finite')
    offsets = sensors[1:] - sensors[0]
    if np.linalg.matrix_rank(offsets) < 2:
        raise ValueError('2D localization requires non-collinear sensors')
    delta_r = c * tdoa
    a = np.column_stack((offsets, delta_r))
    b = 0.5 * (np.sum(sensors[1:] ** 2, axis=1)
               - np.sum(sensors[0] ** 2) - delta_r ** 2)
    estimate = np.linalg.pinv(a) @ b
    return estimate[:2], estimate[2]
