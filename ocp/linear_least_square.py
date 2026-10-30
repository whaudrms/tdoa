"""Linearized TDOA OCP: minimize 0.5 * ||A z - b||**2.

z = [x, y, r0], where r0 is an independent linear variable. No constraint
r0 = ||[x,y] - sensors[0]|| is imposed. With three sensors the system is
underdetermined; the pseudoinverse solver selects a minimum-norm solution.
Inputs use one-way-equivalent TDOA; pulse-echo correction is upstream.
This OCP is fixed for optimization.linear_least_squares.
"""

import numpy as np

from ocp.measurement import validate_measurement


def build_system(sensors, tdoa, c):
    """Return the linear OCP matrix A and target b."""
    sensors, tdoa = validate_measurement(sensors, tdoa, c)
    delta_r = c * tdoa
    a = np.column_stack((sensors[1:] - sensors[0], delta_r))
    b = 0.5 * (np.sum(sensors[1:] ** 2, axis=1)
               - np.sum(sensors[0] ** 2) - delta_r ** 2)
    return a, b


def residual(state, a, b):
    return a @ np.asarray(state, dtype=float) - b


def objective(state, a, b):
    errors = residual(state, a, b)
    return 0.5 * (errors @ errors)
