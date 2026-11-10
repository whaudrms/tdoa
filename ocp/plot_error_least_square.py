"""Plot the measured 2D least-squares objective as a 3D surface.

Run from the project root: python3 -m ocp.plot_error_least_square --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocp.error_least_square import objective


def plot_objective(sensors, delta_r, position, xlim, ylim, points=151):
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
                title='error_least_square objective')
    surface.scatter(*position, objective(position, sensors, delta_r),
                    color='red', s=45, label='Solver estimate')
    contour.contourf(x, y, values, levels=40, cmap='viridis')
    contour.scatter(sensors[:, 0], sensors[:, 1], marker='^',
                    color='white', edgecolors='black', label='Sensors')
    contour.scatter(*sensors[0], marker='*', s=150, color='orange',
                    edgecolors='black', label='Reference sensor')
    contour.scatter(*position, color='red', s=45, label='Solver estimate')
    contour.set(xlabel='x [m]', ylabel='y [m]', title='Objective contours',
                xlim=xlim, ylim=ylim, aspect='equal')
    contour.legend()
    fig.colorbar(artist, ax=[surface, contour], shrink=0.75, label='F(x, y) [m²]')
    return fig


def main():
    from tdoa import SENSORS, run_pipeline
    from optimization.linear_least_squares import SOUND_SPEED

    parser = argparse.ArgumentParser(description=__doc__)
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
    result = run_pipeline(solver='levenberg_marquardt', ocp='error_least_square')
    delta_r = SOUND_SPEED * result['localization_tdoa']
    fig = plot_objective(SENSORS, delta_r, result['position'],
                         args.xlim, args.ylim, args.points)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=180)
    print(f'Saved: {args.save}')
    print(f'Solver estimate [m]: {result["position"]}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
