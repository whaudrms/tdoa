"""Log-barrier continuation with equality-constrained SLSQP inner solves.

Only completed barrier-parameter outer iterations enter history.
"""
import numpy as np
from scipy.optimize import minimize
from ._common import SOUND_SPEED, prepare, geometry, optimality, finish


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp='constrained_least_square',
                      max_iterations=40, tolerance=1e-7, return_history=False):
    z, sensors, a, b = prepare(sensors, tdoa, c, ocp, 'barrier', max_iterations, tolerance)
    history = [z.copy()] if return_history else None
    mu = 1e-3
    for _ in range(max_iterations):
        def fun(state):
            e = a @ state - b
            return 0.5 * (e @ e) - mu * np.log(state[2])
        def jac(state):
            return a.T @ (a @ state - b) - np.array([0., 0., mu / state[2]])
        result = minimize(fun, z, jac=jac, method='SLSQP',
                          bounds=[(None, None), (None, None), (1e-14, None)],
                          constraints=[{'type': 'eq', 'fun': lambda state: geometry(state, sensors)[0],
                                        'jac': lambda state: geometry(state, sensors)[1]}],
                          options={'ftol': 1e-14, 'maxiter': 2000})
        if not result.success:
            raise RuntimeError(f'Barrier inner solve failed: {result.message}')
        z = result.x
        if return_history:
            history.append(z.copy())
        feasibility, stationarity = optimality(z, sensors, a, b)
        if feasibility <= tolerance and stationarity <= tolerance and mu <= tolerance:
            return finish(z, history, return_history)
        mu *= 0.1
    raise RuntimeError('Barrier method did not converge')
