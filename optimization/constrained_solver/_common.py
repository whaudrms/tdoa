"""Shared formulation and validation for nonlinear constrained solvers."""
import numpy as np
from ocp.registry import get_ocp
from optimization.unconstrained_solver.linear_least_squares import SOUND_SPEED, initial_position


def prepare(sensors, tdoa, c, ocp, solver, max_iterations, tolerance):
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    if ocp != 'constrained_least_square':
        raise ValueError(f'{solver} requires constrained_least_square')
    problem = get_ocp(ocp, solver=solver)
    p, r = initial_position(sensors, tdoa, c)
    sensors = np.asarray(sensors, dtype=float)
    a, b = problem.build_system(sensors, tdoa, c)
    return np.r_[p, r], sensors, a, b


def geometry(z, sensors):
    offset = z[:2] - sensors[0]
    distance = np.linalg.norm(offset)
    direction = offset / distance if distance > 0 else np.zeros(2)
    return z[2] - distance, np.r_[-direction, 1.]


def optimality(z, sensors, a, b):
    h, j = geometry(z, sensors)
    g = a.T @ (a @ z - b)
    # Interior range: estimate equality multiplier by least squares.
    lam = -j @ g / (j @ j)
    return max(abs(h), max(0., -z[2])), np.linalg.norm(g + lam * j, np.inf)


def finish(z, history, return_history):
    if not np.isfinite(z).all():
        raise RuntimeError('Solver returned non-finite state')
    result = (z[:2].copy(), float(z[2]))
    return (*result, np.asarray(history)) if return_history else result
