"""Compare all TDOA solvers on one measurement; truth is evaluation-only.

Position/distance errors use one record. Timing shows repeated-call median and IQR.
r0 denotes distance to the first selected (reference) sensor.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from ocp.registry import DEFAULT_OCP, OCP_NAMES, get_ocp, solver_options as ocp_solver_options

from optimization.linear_least_squares import initial_position
from tdoa import SENSORS, SOLVERS, SOUND_SPEED, EXAMPLE_TRUE_POS
from signal_process.tdoa_signal_processing import (
    process_measurement, load_measurement, load_sensor_samples,
    CSV_FILES, SAMPLE_RATE_HZ, RX_COLUMNS,
)
from visualization.waveform_visualization import DATA_DIR


def example_truth(csv_path):
    """Only bundled example recordings may use the bundled ground truth."""
    examples = {(DATA_DIR / f'waveforms_{kind}.csv').resolve()
                for kind in ('clean', 'noisy')}
    if Path(csv_path).resolve() not in examples:
        raise ValueError('Custom CSV requires --true-pos X Y for error evaluation')
    with (DATA_DIR / 'ground_truth.csv').open(newline='') as stream:
        rows = {row['sensor_id']: row for row in csv.DictReader(stream)}
    coordinates = np.array([[float(rows[f'S{i+1}']['sensor_x_m']),
                             float(rows[f'S{i+1}']['sensor_y_m'])]
                            for i in range(len(SENSORS))])
    if not np.allclose(coordinates, SENSORS):
        raise ValueError('Example ground truth sensors do not match SENSORS')
    positions = np.array([[float(rows[f'S{i+1}']['source_x_m']),
                           float(rows[f'S{i+1}']['source_y_m'])]
                          for i in range(len(SENSORS))])
    if not np.isfinite(positions).all() or not np.allclose(positions, positions[0]):
        raise ValueError('Example ground truth source positions are inconsistent')
    return positions[0]


def benchmark_solvers(tdoa, c, lm_lambda, *, warmup=5, repeats=100, seed=42, ocp=DEFAULT_OCP):
    """Time full solver calls, including shared initialization, in shuffled rounds.

    Warmup, validation, RNG, and aggregation are outside the timed region.
    A failed solver has no reported timing statistics, even after partial success.
    """
    for name, value in [('warmup', warmup), ('repeats', repeats)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    rng = np.random.default_rng(seed)
    names = list(SOLVERS)
    samples = {name: [] for name in names}
    failures = {}
    options = {name: ocp_solver_options(name, ocp, lm_lambda)
               for name in names}
    for measured, rounds in [(False, warmup), (True, repeats)]:
        for _ in range(rounds):
            for name in rng.permutation(names):
                if name in failures:
                    continue
                solver = SOLVERS[name]
                kwargs = options[name]
                try:
                    if measured:
                        start = perf_counter()
                        position, r0 = solver(SENSORS, tdoa, c, **kwargs)
                        elapsed_ms = (perf_counter() - start) * 1000
                    else:
                        position, r0 = solver(SENSORS, tdoa, c, **kwargs)
                    if not np.isfinite(position).all() or not np.isfinite(r0):
                        raise RuntimeError('Solver returned non-finite values')
                    if measured:
                        samples[name].append(elapsed_ms)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    failures[name] = str(exc)
    statistics = {}
    for name in names:
        stats = dict(solve_time_ms=np.nan, solve_time_p25_ms=np.nan,
                     solve_time_p75_ms=np.nan, timing_samples=len(samples[name]),
                     timing_warmup=warmup, timing_repeats=repeats, timing_seed=seed,
                     timing_status='ok')
        if name in failures:
            stats['timing_status'] = f'failed: {failures[name]}'
        else:
            q25, median, q75 = np.percentile(samples[name], [25, 50, 75])
            stats.update(solve_time_ms=median, solve_time_p25_ms=q25,
                         solve_time_p75_ms=q75)
        statistics[name] = stats
    return statistics


def compare_solvers(csv_path, true_pos, *, c=SOUND_SPEED, low_hz=30_000,
                    high_hz=50_000, search_range_ms=None, lm_lambda=1e-3,
                    warmup=5, repeats=100, timing_seed=42, round_trip=None, ocp=DEFAULT_OCP):
    """Process once, then independently solve the same measured TDOA vector."""
    problem = get_ocp(ocp)  # Fail clearly before comparing unimplemented objectives.
    true_pos = np.asarray(true_pos, dtype=float)
    if true_pos.shape != (2,) or not np.isfinite(true_pos).all():
        raise ValueError('true_pos must contain two finite coordinates')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('Sound speed must be positive and finite')
    if not np.isfinite(lm_lambda) or lm_lambda < 0:
        raise ValueError('lm_lambda must be non-negative and finite')
    if csv_path is None:
        time_s, raw = load_sensor_samples(CSV_FILES, SAMPLE_RATE_HZ)
    else:
        time_s, raw = load_measurement(csv_path)
    if len(raw) != len(SENSORS):
        raise ValueError('CSV channels must match sensor coordinates')
    processed = process_measurement(time_s, raw, low_hz, high_hz, search_range_ms)
    if round_trip is None:
        round_trip = csv_path is None
    tdoa = processed['tdoa'] / (2.0 if round_trip else 1.0)
    true_r0 = np.linalg.norm(true_pos - SENSORS[0])
    rows = []
    for name, solver in SOLVERS.items():
        row = dict(solver=name,
                   ocp='linear_least_square' if name == 'linear_least_squares' else ocp,
                   evaluation_ocp=ocp, status='ok', x_m=np.nan, y_m=np.nan,
                   position_error_m=np.nan, r0_m=np.nan, true_r0_m=true_r0,
                   r0_signed_error_m=np.nan, r0_absolute_error_m=np.nan,
                   geometric_r0_m=np.nan, geometric_r0_absolute_error_m=np.nan,
                   r0_consistency_gap_m=np.nan, objective_m2=np.nan, solve_time_ms=np.nan)
        try:
            options = ocp_solver_options(name, ocp, lm_lambda)
            position, r0 = solver(SENSORS, tdoa, c, **options)
            if not np.isfinite(position).all() or not np.isfinite(r0):
                raise RuntimeError('Solver returned non-finite values')
            distances = np.linalg.norm(position - SENSORS, axis=1)
            row.update(x_m=position[0], y_m=position[1],
                       position_error_m=np.linalg.norm(position - true_pos),
                       r0_m=r0, r0_signed_error_m=r0 - true_r0,
                       r0_absolute_error_m=abs(r0 - true_r0),
                       geometric_r0_m=distances[0],
                       geometric_r0_absolute_error_m=abs(distances[0] - true_r0),
                       r0_consistency_gap_m=r0 - distances[0],
                       objective_m2=problem.objective(position, SENSORS, c * tdoa))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row['status'] = f'failed: {exc}'
        rows.append(row)
    timing = benchmark_solvers(tdoa, c, lm_lambda, warmup=warmup,
                               repeats=repeats, seed=timing_seed, ocp=ocp)
    for row in rows:
        row.update(timing[row['solver']])
    return rows, tdoa


def plot_comparison(rows, true_pos, tdoa=None, c=SOUND_SPEED):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax, time_ax, error_ax, r0_ax = axes.flat
    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'p', 'h']
    for axis in (ax,):
        axis.scatter(*true_pos, marker='*', s=180, color='black', label='True source', zorder=5)
        axis.set(xlabel='X [m]', ylabel='Y [m]')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=0.25)
    ax.scatter(SENSORS[:, 0], SENSORS[:, 1], marker='^', color='gray', s=70, label='Sensors')
    for i, sensor in enumerate(SENSORS):
        ax.annotate(RX_COLUMNS[i], sensor, xytext=(5, 5), textcoords='offset points')
    for i, row in enumerate(rows):
        if row['status'] != 'ok':
            continue
        for axis in (ax,):
            axis.scatter(row['x_m'], row['y_m'], marker=markers[i % len(markers)],
                         s=100 + i * 45, facecolors='none',
                         edgecolors=colors[i], linewidths=1.8, label=row['solver'], zorder=3)
    ax.set_title('Sensor geometry, TDOA hyperbolas and estimates')
    points = np.vstack((SENSORS, true_pos))
    estimates = [[row['x_m'], row['y_m']] for row in rows
                 if row['status'] == 'ok' and
                 np.isfinite([row['x_m'], row['y_m']]).all()]
    if estimates:
        points = np.vstack((points, estimates))
    lower, upper = points.min(axis=0), points.max(axis=0)
    margin = max(np.max(upper - lower) * 0.35, 0.05)
    ax.set_xlim(lower[0] - margin, upper[0] + margin)
    ax.set_ylim(lower[1] - margin, upper[1] + margin)
    if tdoa is not None:
        # Parameterize the correct hyperbola branch directly. A uniform grid
        # loses the sensor neighborhood when an estimate is extremely distant.
        for i, delay in enumerate(tdoa, 1):
            offset = SENSORS[i] - SENSORS[0]
            baseline = np.linalg.norm(offset)
            difference = c * delay
            if baseline == 0 or abs(difference) >= baseline:
                continue
            direction = offset / baseline
            perpendicular = np.array([-direction[1], direction[0]])
            midpoint = (SENSORS[i] + SENSORS[0]) / 2
            radius = np.max(np.linalg.norm(points - midpoint, axis=1)) + 2 * margin
            if abs(difference) < 1e-12:
                curve = midpoint + np.linspace(-radius, radius, 1000)[:, None] * perpendicular
            else:
                a = abs(difference) / 2
                b = np.sqrt((baseline / 2)**2 - a**2)
                extent = np.arcsinh(radius / b)
                parameter = np.linspace(-extent, extent, 2000)
                along = -np.sign(difference) * a * np.cosh(parameter)
                across = b * np.sinh(parameter)
                curve = (midpoint + along[:, None] * direction
                         + across[:, None] * perpendicular)
            ax.plot(curve[:, 0], curve[:, 1], linewidth=1.1, alpha=0.75,
                    color=colors[i % len(colors)], zorder=1,
                    label=f'{RX_COLUMNS[i]} - {RX_COLUMNS[0]} TDOA')
    ax.legend(fontsize=8)
    labels = [row['solver'].replace('_', '\n') for row in rows]
    indices = np.arange(len(rows))
    times = np.array([row['solve_time_ms'] if row['status'] == 'ok'
                      and row['timing_status'] == 'ok' else np.nan for row in rows])
    bars = time_ax.bar(indices, times, color=colors[:len(rows)])
    valid = np.isfinite(times)
    if valid.any():
        lower = times - np.array([row['solve_time_p25_ms'] for row in rows])
        upper = np.array([row['solve_time_p75_ms'] for row in rows]) - times
        time_ax.errorbar(indices[valid], times[valid],
                         yerr=np.vstack((lower[valid], upper[valid])),
                         fmt='none', ecolor='black', capsize=5, label='25–75% range')
        time_ax.legend(fontsize=8)
    for i in np.flatnonzero(valid):
        time_ax.annotate(f'{times[i]:.3f}',
                         (i, rows[i]['solve_time_p75_ms']), xytext=(0, 5),
                         textcoords='offset points', ha='center', fontsize=9)
    time_ax.set(title=f"Solver time: median of {rows[0]['timing_repeats']} shuffled calls"
                      f" ({rows[0]['timing_warmup']} warmups)",
                ylabel='Full call including initialization [ms]')
    bars = error_ax.bar(indices, [row['position_error_m'] * 1000 for row in rows],
                        color=colors[:len(rows)])
    error_ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
    error_ax.set(title='Position estimation error', ylabel='||estimated position - true position|| [mm]')
    bars = r0_ax.bar(indices, [row['r0_absolute_error_m'] * 1000 for row in rows],
                     label='Returned r0', color='#4477aa')
    r0_ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8)
    r0_ax.set(title='Reference distance error (r0 = distance to reference sensor)',
              ylabel='|estimated r0 - true r0| [mm]')
    r0_ax.legend(fontsize=8)
    for axis in (time_ax, error_ax, r0_ax):
        axis.set_xticks(indices, labels, fontsize=8)
        axis.set_ylim(bottom=0)
        axis.margins(y=0.25)
        axis.grid(axis='y', alpha=0.25)
        for i, row in enumerate(rows):
            if row['status'] != 'ok' or (axis is time_ax and row['timing_status'] != 'ok'):
                axis.text(i, 0.05, 'FAILED', transform=axis.get_xaxis_transform(),
                          ha='center', color='red', fontsize=8)
    fig.suptitle('TDOA solver comparison — single-record errors (not confidence intervals)')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocp', choices=OCP_NAMES, default=DEFAULT_OCP)
    parser.add_argument('--csv', type=Path,
                        help='Combined CSV; defaults to four real sensor sample_1 files')
    parser.add_argument('--propagation', choices=('auto', 'one-way', 'round-trip'),
                        default='auto', help='auto: real sensor files round-trip; combined CSV one-way')
    parser.add_argument('--true-pos', type=float, nargs=2, metavar=('X', 'Y'))
    parser.add_argument('--sound-speed', type=float, default=SOUND_SPEED)
    parser.add_argument('--low-hz', type=float, default=30_000)
    parser.add_argument('--high-hz', type=float, default=50_000)
    parser.add_argument('--search-range-ms', type=float, nargs=2, metavar=('START', 'END'))
    parser.add_argument('--lm-lambda', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=5, help='Warmup calls per solver (default: 5)')
    parser.add_argument('--repeats', type=int, default=100, help='Timed calls per solver (default: 100)')
    parser.add_argument('--timing-seed', type=int, default=42, help='Seed for shuffled solver order')
    parser.add_argument('--output-dir', type=Path, default=Path('results/solver_comparison'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    try:
        if args.true_pos is not None:
            true_pos = np.asarray(args.true_pos)
        elif args.csv is None:
            if EXAMPLE_TRUE_POS is None:
                raise ValueError('Set EXAMPLE_TRUE_POS or provide --true-pos X Y')
            true_pos = np.asarray(EXAMPLE_TRUE_POS, dtype=float)
        else:
            true_pos = example_truth(args.csv)
        round_trip = (args.csv is None) if args.propagation == 'auto' else args.propagation == 'round-trip'
        rows, tdoa = compare_solvers(args.csv, true_pos, c=args.sound_speed,
                                    low_hz=args.low_hz, high_hz=args.high_hz,
                                    search_range_ms=args.search_range_ms, lm_lambda=args.lm_lambda,
                                    warmup=args.warmup, repeats=args.repeats, timing_seed=args.timing_seed,
                                    round_trip=round_trip, ocp=args.ocp)
    except (ValueError, OSError, KeyError, NotImplementedError) as exc:
        parser.error(str(exc))
    print(f'Nonlinear initial position [m]: {initial_position(SENSORS, tdoa, args.sound_speed)[0]}')
    print(f'True position [m]: {true_pos}; true r0 [m]: {rows[0]["true_r0_m"]:.9f}')
    if len(SENSORS) < 4:
        print('linear_least_squares: FAILED; select at least four sensors.')
    if args.csv is None:
        for label, path in zip(RX_COLUMNS, CSV_FILES):
            print(f'Input {label}: {path}')
    print('Nonlinear OCP:', args.ocp, '; linear OCP: linear_least_square')
    print('Propagation:', 'round-trip' if round_trip else 'one-way')
    print(f'Localization TDOA [s] (round-trip corrected when enabled): {tdoa}')
    print(f'{"Solver":<24} {"Position error [mm]":>20} {"r0 error [mm]":>17} {"r0 - geometric [mm]":>23}')
    for row in rows:
        if row['status'] == 'ok':
            print(f'{row["solver"]:<24} {row["position_error_m"]*1000:>20.6f} '
                  f'{row["r0_signed_error_m"]*1000:>+17.6f} {row["r0_consistency_gap_m"]*1000:>+23.6f}')
        else:
            print(f'{row["solver"]}: {row["status"]}')
    print('Timing: full solver calls, including initialization; median [25%, 75%] ms')
    for row in rows:
        print(f"{row['solver']:<24} {row['solve_time_ms']:.6f} "
              f"[{row['solve_time_p25_ms']:.6f}, {row['solve_time_p75_ms']:.6f}] "
              f"n={row['timing_samples']} {row['timing_status']}")
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    fig = plot_comparison(rows, true_pos, tdoa, args.sound_speed)
    try:
        fig.savefig(args.output_dir / 'comparison.png', dpi=160)
        print(f'Saved: {args.output_dir / "comparison.png"}, {args.output_dir / "metrics.csv"}')
        if not args.no_show:
            plt.show()
    finally:
        plt.close(fig)
    if any(row['status'] != 'ok' or row['timing_status'] != 'ok' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
