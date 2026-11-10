"""TDOA least squares using SciPy SLSQP (sequential quadratic programming).

Use the shared initialization and unconstrained distance-difference
objective as the other solvers. No ground truth or position bounds are used.
Inputs must contain one-way-equivalent TDOA; pulse-echo correction is upstream.
"""

import numpy as np
from ocp.registry import DEFAULT_OCP, get_ocp
from scipy.optimize import minimize

from optimization.unconstrained_solver.linear_least_squares import (
    SOUND_SPEED, initial_position,
)


def estimate_position(sensors, tdoa, c=SOUND_SPEED, *, ocp=DEFAULT_OCP,
                      max_iterations=10_000, ftol=1e-12, return_history=False):
    """Return (position, distance to reference), or raise on SLSQP failure.

    ftol controls SLSQP's objective, step and optimality stopping checks;
    success does not guarantee a global minimum.
    return_history=True appends an array containing the initial state and
    SLSQP major-iteration states (not line-search trial points).
    """
    problem = get_ocp(ocp, solver='sqp')
    if (not isinstance(max_iterations, int) or isinstance(max_iterations, bool)
            or max_iterations < 1):
        raise ValueError('max_iterations must be a positive integer')
    if not np.isfinite(ftol) or ftol <= 0:
        raise ValueError('ftol must be positive and finite')
    initial, initial_r0 = initial_position(sensors, tdoa, c)
    sensors = np.asarray(sensors, dtype=float)
    delta_r = c * np.asarray(tdoa, dtype=float)
    constrained = hasattr(problem, 'constraints')
    if constrained:
        initial = np.r_[initial, initial_r0]
    history = [initial.copy()] if return_history else None

    def record_iteration(state):
        history.append(np.array(state, copy=True))

    result = minimize(problem.objective_gradient, initial, args=(sensors, delta_r),
                      method='SLSQP', jac=True,
                      callback=record_iteration if return_history else None,
                      constraints=problem.constraints(sensors) if constrained else (),
                      options={'maxiter': max_iterations, 'ftol': ftol})
    if not result.success:
        raise RuntimeError(f'SQP did not converge: {result.message}')
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError('SQP returned non-finite values')
    if constrained:
        if (result.x[2] < -problem.CONSTRAINT_TOLERANCE or
                abs(problem.range_consistency(result.x, sensors)) > problem.CONSTRAINT_TOLERANCE):
            raise RuntimeError('SQP result violates the reference range constraints')
        estimate = (result.x[:2], result.x[2])
    else:
        estimate = (result.x, np.linalg.norm(result.x - sensors[0]))
    if return_history:
        if not np.array_equal(history[-1], result.x):
            history.append(result.x.copy())
        return (*estimate, np.asarray(history))
    return estimate
