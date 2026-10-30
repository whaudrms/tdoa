"""Linearized least squares in (x, y, r0) with physical range constraints.

r0**2 = ||p-sensors[0]||**2 and r0 >= 0.
SQP enforces the equivalent r0 - ||p-sensors[0]|| = 0 equality, avoiding
the squared equality's zero Jacobian at the reference initial position.
The norm uses a zero subgradient at that initial position.
"""

import numpy as np

IMPLEMENTED = True
SUPPORTED_SOLVERS = ('sqp',)
CONSTRAINT_TOLERANCE = 1e-7

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


def objective_gradient(state, sensors, delta_r):
    a, b = build_system(sensors, delta_r, 1.0)
    errors = residual(state, a, b)
    return 0.5 * (errors @ errors), a.T @ errors


def objective(position_or_state, sensors, delta_r):
    """Evaluate a state, or lift a position onto the range constraint."""
    state = np.asarray(position_or_state, dtype=float)
    if state.shape == (2,):
        state = np.r_[state, np.linalg.norm(state - sensors[0])]
    return objective_gradient(state, sensors, delta_r)[0]


def range_equality(state, sensors):
    offset = state[:2] - sensors[0]
    return state[2] ** 2 - offset @ offset


def range_consistency(state, sensors):
    return state[2] - np.linalg.norm(state[:2] - sensors[0])


def range_consistency_jacobian(state, sensors):
    offset = state[:2] - sensors[0]
    distance = np.linalg.norm(offset)
    direction = offset / distance if distance > 0 else np.zeros(2)
    return np.r_[-direction, 1.0]


def constraints(sensors):
    return (
        {'type': 'eq', 'fun': range_consistency,
         'jac': range_consistency_jacobian, 'args': (sensors,)},
        {'type': 'ineq', 'fun': lambda state: state[2],
         'jac': lambda state: np.array([0., 0., 1.])},
    )

