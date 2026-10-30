"""Explicit OCP selection for nonlinear solvers.

Objective providers expose objective, objective_gradient,
objective_gradient_hessian and gauss_newton_terms as needed by each solver.
Future constrained problems also need constraint-capable solver integration;
the placeholder must remain unavailable until that is implemented.
"""

from importlib import import_module

OCP_NAMES = ('error_least_square', 'noise_least_square', 'constrained_least_square')
DEFAULT_OCP = 'error_least_square'


def get_ocp(name=DEFAULT_OCP, solver=None):
    if name not in OCP_NAMES:
        raise ValueError(f'Unknown OCP: {name}; choose from {OCP_NAMES}')
    module = import_module(f'ocp.{name}')
    if not module.IMPLEMENTED:
        raise NotImplementedError(f'OCP {name} is not implemented yet')
    if solver is not None and solver not in getattr(module, 'SUPPORTED_SOLVERS', (solver,)):
        raise ValueError(f'OCP {name} supports only {module.SUPPORTED_SOLVERS}; '
                         f'{solver} cannot enforce its constraints')
    return module


def solver_options(solver, ocp=DEFAULT_OCP, lm_lambda=1e-3):
    """The linear solver always retains its dedicated linear OCP."""
    if solver == 'linear_least_squares':
        return {}
    options = {'ocp': ocp}
    if solver == 'levenberg_marquardt':
        options['damping'] = lm_lambda
    return options
