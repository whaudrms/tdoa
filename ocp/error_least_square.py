"""Unconstrained 2D distance-difference least-squares problem.

F(p) = 0.5 * sum_i (||p-s_i|| - ||p-s_0|| - delta_r_i)**2.
delta_r is in meters (sound speed times one-way-equivalent TDOA).
Pulse-echo correction belongs upstream. Ground truth is never an input.
Solvers supply their own update rules, initialization and stopping criteria.
Distances use sqrt(||p-s||**2 + epsilon**2), epsilon = 1 nm, so all solvers
can start exactly at the reference sensor without undefined derivatives.
"""

import numpy as np

DISTANCE_EPS_M = 1e-9
IMPLEMENTED = True


def _distances(offsets):
    return np.sqrt(np.sum(offsets * offsets, axis=1) + DISTANCE_EPS_M**2)


def residual(position, sensors, delta_r):
    """Distance-difference residuals; defined even at a sensor position."""
    distances = _distances(np.asarray(position) - sensors)
    return distances[1:] - distances[0] - delta_r


def objective(position, sensors, delta_r):
    errors = residual(position, sensors, delta_r)
    return 0.5 * (errors @ errors)


def residual_jacobian(position, sensors, delta_r):
    offsets = np.asarray(position) - sensors
    distances = _distances(offsets)
    errors = distances[1:] - distances[0] - delta_r
    directions = offsets / distances[:, None]
    return errors, directions[1:] - directions[0]


def objective_gradient(position, sensors, delta_r):
    errors, jacobian = residual_jacobian(position, sensors, delta_r)
    return 0.5 * (errors @ errors), jacobian.T @ errors


def gauss_newton_terms(position, sensors, delta_r):
    """Objective, gradient, approximate Hessian and residual Jacobian."""
    errors, jacobian = residual_jacobian(position, sensors, delta_r)
    return (0.5 * (errors @ errors), jacobian.T @ errors,
            jacobian.T @ jacobian, jacobian)


def objective_gradient_hessian(position, sensors, delta_r):
    """Objective, gradient and exact Hessian of the smoothed objective."""
    errors, jacobian = residual_jacobian(position, sensors, delta_r)
    offsets = np.asarray(position) - sensors
    distances = _distances(offsets)
    directions = offsets / distances[:, None]
    distance_hessians = (
        np.eye(2) - directions[:, :, None] * directions[:, None, :]
    ) / distances[:, None, None]
    hessian = jacobian.T @ jacobian + np.einsum(
        'i,ijk->jk', errors, distance_hessians[1:] - distance_hessians[0])
    return 0.5 * (errors @ errors), jacobian.T @ errors, hessian
