"""Plot the measured 2D least-squares objective as a 3D surface.

Run from the project root: python3 -m visualization.plot_error_least_square --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocp.error_least_square import objective


def plot_objective(sensors, delta_r, position, xlim, ylim, points=151, history=None,
                   solver='gradient_descent'):
    """Evaluate the provider itself so the plot matches the solver objective."""
    x, y = np.meshgrid(np.linspace(*xlim, points), np.linspace(*ylim, points))
    values = np.array([
        objective(p, sensors, delta_r)
        for p in np.column_stack((x.ravel(), y.ravel()))
    ]).reshape(x.shape)
    fig = plt.figure(figsize=(13, 5), layout='constrained')
    surface = fig.add_subplot(121, projection='3d')
    contour = fig.add_subplot(122)
    artist = surface.plot_surface(x, y, values, cmap='viridis',
                                  rcount=points, ccount=points, alpha=0.9)
    surface.set(xlabel='x [m]', ylabel='y [m]', zlabel='F(x, y) [m²]',
                title=f'error_least_square objective ({solver})')
    surface.scatter(*position, objective(position, sensors, delta_r),
                    color='red', s=45, label='Solver estimate')
    contour.contourf(x, y, values, levels=40, cmap='viridis')
    contour.scatter(sensors[:, 0], sensors[:, 1], marker='^',
                    color='white', edgecolors='black', label='Sensors')
    contour.scatter(*sensors[0], marker='*', s=150, color='orange',
                    edgecolors='black', label='Reference sensor')
    contour.scatter(*position, color='red', s=45, label='Solver estimate')
    if history is not None:
        costs = np.array([objective(p, sensors, delta_r) for p in history])
        surface.plot(history[:, 0], history[:, 1], costs, 'o-',
                     color='orangered', markersize=3, label=f'{solver} iterations')
        contour.plot(history[:, 0], history[:, 1], 'o-', color='orangered',
                     markersize=3, label=f'{solver} iterations')
        surface.scatter(*history[0], costs[0], marker='s', color='cyan',
                        edgecolors='black', s=50, label='Initial state')
        contour.scatter(*history[0], marker='s', color='cyan',
                        edgecolors='black', s=50, label='Initial state')
        for iteration in range(0, max(1, len(history) - 3), 3):
            contour.annotate(str(iteration), history[iteration],
                             xytext=(4, 5), textcoords='offset points', fontsize=8)
        surface.legend()
    contour.set(xlabel='x [m]', ylabel='y [m]', title='Objective contours',
                xlim=xlim, ylim=ylim, aspect='equal')
    contour.legend()
    fig.colorbar(artist, ax=[surface, contour], shrink=0.75, label='F(x, y) [m²]')
    return fig


def main():
    from tdoa import SENSORS, SOLVERS, run_pipeline
    from optimization.unconstrained_solver.linear_least_squares import SOUND_SPEED
    from ocp.registry import solver_options

    parser = argparse.ArgumentParser(description=__doc__)
    # The linear solver optimizes a different OCP, so it is excluded here.
    parser.add_argument('--solver', choices=(
        'gradient_descent', 'newton_method', 'gauss_newton', 'levenberg_marquardt', 'sqp'),
        default='gradient_descent', help='Solver for error_least_square')
    parser.add_argument('--lm-lambda', type=float, default=1e-3,
                        help='Fixed LM damping (default: 0.001)')
    parser.add_argument('--xlim', nargs=2, type=float, default=(-0.1, 0.6))
    parser.add_argument('--ylim', nargs=2, type=float, default=(-0.1, 0.5))
    parser.add_argument('--points', type=int, default=151)
    parser.add_argument('--save', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'error_least_square_3d.png')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    for bounds in (args.xlim, args.ylim):
        if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            parser.error('Axis limits must be finite and increasing')
    if args.points < 2:
        parser.error('--points must be at least 2')
    if not np.isfinite(args.lm_lambda) or args.lm_lambda < 0:
        parser.error('--lm-lambda must be non-negative and finite')
    result = run_pipeline(solver=args.solver, ocp='error_least_square',
                          lm_lambda=args.lm_lambda)
    delta_r = SOUND_SPEED * result['localization_tdoa']
    position, _, history = SOLVERS[args.solver](
        SENSORS, result['localization_tdoa'], SOUND_SPEED,
        **solver_options(args.solver, 'error_least_square', args.lm_lambda),
        return_history=True)
    fig = plot_objective(SENSORS, delta_r, position,
                         args.xlim, args.ylim, args.points, history=history,
                         solver=args.solver)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=180)
    print(f'Saved: {args.save}')
    print(f'Solver: {args.solver}')
    print(f'Solver estimate [m]: {result["position"]}')
    print(f'Iterations: {len(history) - 1}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
