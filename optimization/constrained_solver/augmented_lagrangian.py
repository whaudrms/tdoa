"""Equality augmented Lagrangian with a nonnegative range bound."""
import numpy as np
from scipy.optimize import least_squares
from ._common import SOUND_SPEED, prepare, geometry, optimality, finish


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp='constrained_least_square',
                      max_iterations=100, tolerance=1e-7, return_history=False):
    z, sensors, a, b = prepare(sensors, tdoa, c, ocp, 'augmented_lagrangian', max_iterations, tolerance)
    history = [z.copy()] if return_history else None
    rho, lam, previous = 1., 0., np.inf
    for _ in range(max_iterations):
        scale = np.sqrt(rho)
        def fun(state):
            h, _ = geometry(state, sensors)
            return np.r_[a @ state - b, scale * (h + lam / rho)]
        def jac(state):
            return np.vstack((a, scale * geometry(state, sensors)[1]))
        result = least_squares(fun, z, jac=jac, bounds=([-np.inf, -np.inf, 0.], np.inf),
                               ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=2000)
        z = result.x
        h, _ = geometry(z, sensors)
        lam += rho * h
        if return_history:
            history.append(z.copy())
        feasibility, stationarity = optimality(z, sensors, a, b)
        if feasibility <= tolerance and stationarity <= tolerance:
            return finish(z, history, return_history)
        if abs(h) > 0.25 * previous:
            rho *= 10.
        previous = abs(h)
    raise RuntimeError('Augmented Lagrangian did not converge')
