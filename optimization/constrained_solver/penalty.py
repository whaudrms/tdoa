"""Quadratic penalty continuation; history records penalty outer iterations."""
import numpy as np
from scipy.optimize import least_squares
from ._common import SOUND_SPEED, prepare, geometry, optimality, finish


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp='constrained_least_square',
                      max_iterations=40, tolerance=1e-7, return_history=False):
    z, sensors, a, b = prepare(sensors, tdoa, c, ocp, 'penalty', max_iterations, tolerance)
    history = [z.copy()] if return_history else None
    rho = 1.
    for _ in range(max_iterations):
        scale = np.sqrt(rho)
        def fun(state):
            h, _ = geometry(state, sensors)
            return np.r_[a @ state - b, scale * h, scale * min(0., state[2])]
        def jac(state):
            _, j = geometry(state, sensors)
            return np.vstack((a, scale * j, [0., 0., scale if state[2] < 0 else 0.]))
        result = least_squares(fun, z, jac=jac, ftol=1e-13, xtol=1e-13,
                               gtol=1e-13, max_nfev=2000)
        z = result.x
        if return_history:
            history.append(z.copy())
        feasibility, stationarity = optimality(z, sensors, a, b)
        if feasibility <= tolerance and stationarity <= tolerance:
            return finish(z, history, return_history)
        rho *= 10.
    raise RuntimeError('Penalty method did not converge')
