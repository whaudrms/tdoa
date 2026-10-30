"""Common input validation, independent of the selected objective."""

import numpy as np


def validate_measurement(sensors, tdoa, c):
    sensors = np.asarray(sensors, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float)
    if sensors.ndim != 2 or sensors.shape[1] != 2 or len(sensors) < 3:
        raise ValueError('At least three 2D sensor coordinates are required')
    if tdoa.shape != (len(sensors) - 1,):
        raise ValueError('Provide one TDOA value per non-reference sensor')
    if not np.isfinite(sensors).all() or not np.isfinite(tdoa).all():
        raise ValueError('Sensor coordinates and TDOA must be finite')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('Sound speed must be positive and finite')
    if np.linalg.matrix_rank(sensors[1:] - sensors[0]) < 2:
        raise ValueError('2D localization requires non-collinear sensors')
    return sensors, tdoa
