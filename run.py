"""Compare all TDOA solvers on one measurement; truth is evaluation-only.

Position/distance errors use one record. Timing shows repeated-call median and IQR.
d_0 denotes distance to S1 (the value called r1 by the solvers).
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from tdoa import SENSORS, SOLVERS, SOUND_SPEED
from signal_process.tdoa_signal_processing import process_measurement, load_measurement
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


def benchmark_solvers(tdoa, c, lm_lambda, *, warmup=5, repeats=100, seed=42):
    """Time full solver calls, including linear initialization, in shuffled rounds.

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
    options = {name: {'damping': lm_lambda} if name == 'levenberg_marquardt' else {}
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
                        position, d0 = solver(SENSORS, tdoa, c, **kwargs)
                        elapsed_ms = (perf_counter() - start) * 1000
                    else:
                        position, d0 = solver(SENSORS, tdoa, c, **kwargs)
                    if not np.isfinite(position).all() or not np.isfinite(d0):
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
                    warmup=5, repeats=100, timing_seed=42):
    """Process once, then independently solve the same measured TDOA vector."""
    true_pos = np.asarray(true_pos, dtype=float)
    if true_pos.shape != (2,) or not np.isfinite(true_pos).all():
        raise ValueError('true_pos must contain two finite coordinates')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('Sound speed must be positive and finite')
    if not np.isfinite(lm_lambda) or lm_lambda < 0:
        raise ValueError('lm_lambda must be non-negative and finite')
    time_s, raw = load_measurement(csv_path)
    if len(raw) != len(SENSORS):
        raise ValueError('CSV channels must match sensor coordinates')
    processed = process_measurement(time_s, raw, low_hz, high_hz, search_range_ms)
    tdoa = processed['tdoa']
    true_d0 = np.linalg.norm(true_pos - SENSORS[0])
    rows = []
    for name, solver in SOLVERS.items():
        row = dict(solver=name, status='ok', x_m=np.nan, y_m=np.nan,
                   position_error_m=np.nan, d0_m=np.nan, true_d0_m=true_d0,
                   d0_signed_error_m=np.nan, d0_absolute_error_m=np.nan,
                   geometric_d0_m=np.nan, geometric_d0_absolute_error_m=np.nan,
                   d0_consistency_gap_m=np.nan, objective_m2=np.nan, solve_time_ms=np.nan)
        try:
            options = {'damping': lm_lambda} if name == 'levenberg_marquardt' else {}
            position, d0 = solver(SENSORS, tdoa, c, **options)
            if not np.isfinite(position).all() or not np.isfinite(d0):
                raise RuntimeError('Solver returned non-finite values')
            distances = np.linalg.norm(position - SENSORS, axis=1)
            residual = distances[1:] - distances[0] - c * tdoa
            row.update(x_m=position[0], y_m=position[1],
                       position_error_m=np.linalg.norm(position - true_pos),
                       d0_m=d0, d0_signed_error_m=d0 - true_d0,
                       d0_absolute_error_m=abs(d0 - true_d0),
                       geometric_d0_m=distances[0],
                       geometric_d0_absolute_error_m=abs(distances[0] - true_d0),
                       d0_consistency_gap_m=d0 - distances[0],
                       objective_m2=0.5 * (residual @ residual))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row['status'] = f'failed: {exc}'
        rows.append(row)
    timing = benchmark_solvers(tdoa, c, lm_lambda, warmup=warmup,
                               repeats=repeats, seed=timing_seed)
    for row in rows:
        row.update(timing[row['solver']])
    return rows, tdoa


def plot_comparison(rows, true_pos):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax, time_ax, error_ax, d0_ax = axes.flat
    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'p']
    for axis in (ax,):
        axis.scatter(*true_pos, marker='*', s=180, color='black', label='True source', zorder=5)
        axis.set(xlabel='X [m]', ylabel='Y [m]')
        axis.set_aspect('equal', adjustable='datalim')
        axis.grid(alpha=0.25)
    ax.scatter(SENSORS[:, 0], SENSORS[:, 1], marker='^', color='gray', s=70, label='Sensors')
    for i, sensor in enumerate(SENSORS):
        ax.annotate(f'S{i+1}', sensor, xytext=(5, 5), textcoords='offset points')
    for i, row in enumerate(rows):
        if row['status'] != 'ok':
            continue
        for axis in (ax,):
            axis.scatter(row['x_m'], row['y_m'], marker=markers[i % len(markers)],
                         s=100 + i * 45, facecolors='none',
                         edgecolors=colors[i], linewidths=1.8, label=row['solver'], zorder=3)
    ax.set_title('Sensor geometry and estimates')
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
    bars = d0_ax.bar(indices, [row['d0_absolute_error_m'] * 1000 for row in rows],
                     label='Returned d_0', color='#4477aa')
    d0_ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8)
    d0_ax.set(title='Reference distance error (d_0 = distance to S1)',
              ylabel='|estimated d_0 - true d_0| [mm]')
    d0_ax.legend(fontsize=8)
    for axis in (time_ax, error_ax, d0_ax):
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
    parser.add_argument('--csv', type=Path, default=DATA_DIR / 'waveforms_noisy.csv')
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
        true_pos = np.asarray(args.true_pos) if args.true_pos is not None else example_truth(args.csv)
        rows, tdoa = compare_solvers(args.csv, true_pos, c=args.sound_speed,
                                    low_hz=args.low_hz, high_hz=args.high_hz,
                                    search_range_ms=args.search_range_ms, lm_lambda=args.lm_lambda,
                                    warmup=args.warmup, repeats=args.repeats, timing_seed=args.timing_seed)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f'True position [m]: {true_pos}; true d_0 [m]: {rows[0]["true_d0_m"]:.9f}')
    print(f'Measured TDOA [s]: {tdoa}')
    print(f'{"Solver":<24} {"Position error [mm]":>20} {"d_0 error [mm]":>17} {"d_0 - geometric [mm]":>23}')
    for row in rows:
        if row['status'] == 'ok':
            print(f'{row["solver"]:<24} {row["position_error_m"]*1000:>20.6f} '
                  f'{row["d0_signed_error_m"]*1000:>+17.6f} {row["d0_consistency_gap_m"]*1000:>+23.6f}')
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
    fig = plot_comparison(rows, true_pos)
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
