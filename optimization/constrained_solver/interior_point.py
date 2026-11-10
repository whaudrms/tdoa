"""Primal-dual central-path method with bounded least-squares KKT inner solves.

Unknowns are (x, y, r0, equality multiplier, inequality multiplier).
History contains states only after each barrier-parameter outer iteration.
"""
import numpy as np
from scipy.optimize import least_squares
from ._common import SOUND_SPEED, prepare, geometry, finish


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp='constrained_least_square',
                      max_iterations=40, tolerance=1e-7, return_history=False):
    z, sensors, a, b = prepare(sensors, tdoa, c, ocp, 'interior_point', max_iterations, tolerance)
    history = [z.copy()] if return_history else None
    mu = 1e-3
    state = np.r_[z, 0., mu / z[2]]
    for _ in range(max_iterations):
        def equations(w):
            h, j = geometry(w[:3], sensors)
            stationarity = a.T @ (a @ w[:3] - b) + w[3] * j - np.array([0., 0., w[4]])
            return np.r_[stationarity, h, w[2] * w[4] / mu - 1.]
        result = least_squares(equations, state,
                               bounds=([-np.inf, -np.inf, 1e-14, -np.inf, 1e-14], np.inf),
                               ftol=1e-13, xtol=1e-13, gtol=1e-13, x_scale='jac', max_nfev=3000)
        state = result.x
        z = state[:3]
        if return_history:
            history.append(z.copy())
        residual = np.linalg.norm(equations(state), np.inf)
        if residual <= tolerance and state[2] * state[4] <= tolerance:
            return finish(z, history, return_history)
        if not result.success or residual > max(tolerance, mu * 0.1):
            raise RuntimeError('Interior-point KKT inner solve did not converge')
        mu *= 0.1
    raise RuntimeError('Interior-point method did not converge')
