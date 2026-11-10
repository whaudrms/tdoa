"""Plot the constrained linear least-squares objective on its feasible surface.

At each (x, y), r0 is set to the geometric reference range, satisfying
r0 = ||position - sensors[0]|| and r0 >= 0.
Run from the project root:
python3 -m visualization.plot_constrained_least_square --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocp.constrained_least_square import objective


def plot_objective(sensors, delta_r, position, xlim, ylim, points=151, history=None, solver='sqp'):
    """Evaluate F(x, y, r0(x, y)) using the provider's position lifting."""
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
    surface.set(xlabel='x [m]', ylabel='y [m]', zlabel='F(x, y, r0(x, y)) [m⁴]',
                title='constrained_least_square (feasible surface)')
    surface.scatter(*position, objective(position, sensors, delta_r),
                    color='red', s=45, label='Solver estimate')
    contour.contourf(x, y, values, levels=40, cmap='viridis')
    contour.scatter(sensors[:, 0], sensors[:, 1], marker='^',
                    color='white', edgecolors='black', label='Sensors')
    contour.scatter(*sensors[0], marker='*', s=150, color='orange',
                    edgecolors='black', label='Reference sensor')
    contour.scatter(*position, color='red', s=45, label='Solver estimate')
    if history is not None:
        costs = [objective(state, sensors, delta_r) for state in history]
        surface.plot(history[:, 0], history[:, 1], costs, 'o-',
                     color='orangered', markersize=3, label=f'{solver} outer iterations (actual F)')
        contour.plot(history[:, 0], history[:, 1], 'o-', color='orangered',
                     markersize=3, label=f'{solver} outer iteration projection')
        contour.scatter(*history[0, :2], marker='s', color='cyan',
                        edgecolors='black', label='Initial state')
        surface.legend()
    contour.set(xlabel='x [m]', ylabel='y [m]', title='Objective contours',
                xlim=xlim, ylim=ylim, aspect='equal')
    contour.legend()
    fig.colorbar(artist, ax=[surface, contour], shrink=0.75,
                 label='F(x, y, r0(x, y)) [m⁴]')
    return fig


def plot_feasible_set(sensors, delta_r, position, r0, xlim, ylim, points=151,
                      history=None, solver='sqp'):
    """Show feasible states in 3D, colored by their objective values."""
    x, y = np.meshgrid(np.linspace(*xlim, points), np.linspace(*ylim, points))
    ranges = np.sqrt((x - sensors[0, 0])**2 + (y - sensors[0, 1])**2)
    values = np.array([
        objective(state, sensors, delta_r)
        for state in np.column_stack((x.ravel(), y.ravel(), ranges.ravel()))
    ]).reshape(x.shape)
    colors = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=float(values.min()), vmax=float(values.max())),
        cmap='viridis')
    fig = plt.figure(figsize=(9, 7), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, ranges, facecolors=colors.to_rgba(values),
                    rcount=points, ccount=points, shade=False,
                    linewidth=0, antialiased=True, alpha=0.65)
    # Constant-range rings emphasize that only the surface is feasible.
    max_range = float(ranges.max())
    ax.contour(x, y, ranges, levels=np.linspace(0, max_range, 9)[1:],
               colors='navy', linewidths=0.7)
    ax.scatter(*sensors[0], 0, marker='*', s=160, color='orange',
               edgecolors='black', label='Cone apex: reference sensor')
    ax.scatter(*position, r0, color='red', s=55, depthshade=False,
               label=f'{solver} estimate')
    if history is not None:
        ax.plot(history[:, 0], history[:, 1], history[:, 2], 'o-',
                color='orangered', markersize=4, label=f'{solver} outer iterations (may be infeasible)')
        ax.scatter(*history[0], marker='s', color='cyan', edgecolors='black',
                   s=50, label='Initial state')
        for iteration, state in enumerate(history):
            if iteration % 3 == 0 and iteration < len(history) - 3:
                ax.text(*state, str(iteration), fontsize=8)
    ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='r0 [m]',
           xlim=xlim, ylim=ylim,
           zlim=(min(0, float(history[:, 2].min())) if history is not None else 0,
                 max(max_range, float(r0),
                     float(history[:, 2].max()) if history is not None else 0)),
           title='Feasible set colored by objective F(x, y, r0)\n'
                 'r0 = ||(x, y) - s0||; x/y limits are display bounds')
    ax.legend(loc='upper left')
    fig.colorbar(colors, ax=ax, shrink=0.7, pad=0.12,
                 label='F(x, y, r0) [m⁴]')
    return fig


def main():
    from tdoa import SENSORS, SOLVERS, run_pipeline
    from optimization.unconstrained_solver.linear_least_squares import SOUND_SPEED
    from ocp.constrained_least_square import SUPPORTED_SOLVERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=SUPPORTED_SOLVERS, default='sqp')
    parser.add_argument('--xlim', nargs=2, type=float, default=(-0.1, 0.6))
    parser.add_argument('--ylim', nargs=2, type=float, default=(-0.1, 0.5))
    parser.add_argument('--points', type=int, default=151)
    parser.add_argument('--save', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'constrained_least_square_3d.png')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-feasible', type=Path,
                        help='Feasible-set image; defaults beside --save with _feasible suffix')
    args = parser.parse_args()
    for bounds in (args.xlim, args.ylim):
        if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            parser.error('Axis limits must be finite and increasing')
    if args.points < 2:
        parser.error('--points must be at least 2')
    result = run_pipeline(solver=args.solver, ocp='constrained_least_square')
    delta_r = SOUND_SPEED * result['localization_tdoa']
    position, r0, history = SOLVERS[args.solver](
        SENSORS, result['localization_tdoa'], SOUND_SPEED,
        ocp='constrained_least_square', return_history=True)
    print(f'{args.solver} path: initial state + {len(history) - 1} iteration states')
    fig = plot_objective(SENSORS, delta_r, result['position'],
                         args.xlim, args.ylim, args.points, history=history, solver=args.solver)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=180)
    print(f'Saved: {args.save}')
    print(f'Solver estimate [m]: {result["position"]}')
    feasible_fig = plot_feasible_set(SENSORS, delta_r, position, r0,
                                    args.xlim, args.ylim, args.points, history=history, solver=args.solver)
    feasible_path = args.save_feasible or args.save.with_name(
        args.save.stem + '_feasible' + args.save.suffix)
    feasible_path.parent.mkdir(parents=True, exist_ok=True)
    feasible_fig.savefig(feasible_path, dpi=180)
    print(f'Saved feasible set: {feasible_path}')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    plt.close(feasible_fig)


if __name__ == '__main__':
    main()
