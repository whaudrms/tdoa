"""Linearized 2D TDOA localization using a pseudoinverse."""

import numpy as np

# Fixed OCP: this solver does not use the nonlinear objective selection.
from ocp.linear_least_square import build_system
from ocp.measurement import validate_measurement


SOUND_SPEED = 343.0  # m/s


def estimate_position(sensors, tdoa, c=SOUND_SPEED):
    """Solve A @ [x, y, r0] = b using reference-relative differences.

    At least four sensors are required for the three linear unknowns.
    The returned r0
    is an independent linear variable and may disagree with geometric range.
    """
    sensors, tdoa = validate_measurement(sensors, tdoa, c)
    if len(sensors) < 4:
        raise ValueError('linear_least_squares requires at least four sensors; '
                         'select four or more sensors')
    a, b = build_system(sensors, tdoa, c)
    estimate = np.linalg.pinv(a) @ b
    return estimate[:2], estimate[2]


def initial_position(sensors, tdoa, c=SOUND_SPEED):
    """Use the sensor centroid, with a shared offset if it hits a sensor."""
    sensors, _ = validate_measurement(sensors, tdoa, c)
    position = np.mean(sensors, axis=0)
    scale = max(float(np.max(np.ptp(sensors, axis=0))), 1e-3)
    clearance = scale * 1e-6
    # Deterministic offset, identical for every solver; no ground truth used.
    while np.any(np.linalg.norm(position - sensors, axis=1) <= clearance):
        position = position + clearance * np.array([1.0, 0.5])
    return position, np.linalg.norm(position - sensors[0])
