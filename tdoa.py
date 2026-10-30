"""Run waveform processing, envelope cross-correlation TDOA localization, and plots."""

import argparse
from pathlib import Path

import numpy as np
from ocp.registry import DEFAULT_OCP, OCP_NAMES, solver_options as ocp_solver_options

from optimization.linear_least_squares import SOUND_SPEED, estimate_position
from optimization.gradient_descent import estimate_position as estimate_gradient_descent
from optimization.newton_method import estimate_position as estimate_newton
from optimization.gauss_newton import estimate_position as estimate_gauss_newton
from optimization.levenberg_marquardt import estimate_position as estimate_lm
from optimization.sqp import estimate_position as estimate_sqp
from signal_process.tdoa_signal_processing import (
    plot_processed, print_peak_times, process_measurement, load_measurement,
    load_sensor_samples, CSV_FILES, SAMPLE_RATE_HZ, RX_COLUMNS, SENSOR_COORDINATES,
)
from visualization.tdoa_visualization import visualize_tdoa
from visualization.waveform_visualization import DATA_DIR


# Sensor coordinates [m], in rx0, rx1, rx2, rx3 order. rx0 is the reference.
SENSORS = np.asarray(SENSOR_COORDINATES, dtype=float)
SOLVERS = {'linear_least_squares': estimate_position,
           'gradient_descent': estimate_gradient_descent,
           'newton_method': estimate_newton,
           'gauss_newton': estimate_gauss_newton,
           'levenberg_marquardt': estimate_lm,
           'sqp': estimate_sqp}
EXAMPLE_TRUE_POS = [0.40,0.20]


def run_pipeline(csv_path=None, sensors=SENSORS, c=SOUND_SPEED,
                 low_hz=30_000, high_hz=50_000, search_range_ms=None,
                 solver='linear_least_squares', lm_lambda=1e-3, round_trip=None, ocp=DEFAULT_OCP):
    """Process measured waveforms without using ground-truth positions or times."""
    if solver not in SOLVERS:
        raise ValueError(f'Unknown solver: {solver}; choose from {tuple(SOLVERS)}')
    if csv_path is None:
        time_s, raw = load_sensor_samples(CSV_FILES, SAMPLE_RATE_HZ)
    else:
        time_s, raw = load_measurement(csv_path)
    expected = RX_COLUMNS.copy()
    if len(raw) != len(sensors):
        raise ValueError('CSV channels must match sensor coordinates')
    processed = process_measurement(time_s, raw, low_hz, high_hz, search_range_ms)
    # Separate real sensor captures are pulse-echo; combined legacy CSVs are one-way.
    if round_trip is None:
        round_trip = csv_path is None
    tdoa = processed['tdoa'] / (2.0 if round_trip else 1.0)
    solver_options = ocp_solver_options(solver, ocp, lm_lambda)
    position, r0 = SOLVERS[solver](sensors, tdoa, c, **solver_options)
    return dict(time_s=time_s, raw=raw, labels=expected, **processed,
                position=position, r0=r0, solver=solver,
                localization_tdoa=tdoa, round_trip=round_trip)


def main():
    results_dir = Path(__file__).resolve().parent / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path,
                        help='Combined CSV; defaults to the four real sensor sample_1 files')
    parser.add_argument('--propagation', choices=('auto', 'one-way', 'round-trip'),
                        default='auto', help='auto: real sensor files round-trip; combined CSV one-way')
    parser.add_argument('--ocp', choices=OCP_NAMES, default=DEFAULT_OCP,
                        help='Nonlinear OCP; linear solver always uses linear_least_square')
    parser.add_argument('--solver', choices=tuple(SOLVERS),
                        default='linear_least_squares', help='Position estimation method')
    parser.add_argument('--lm-lambda', type=float, default=1e-3,
                        help='Fixed non-negative damping for levenberg_marquardt (default: 0.001)')
    parser.add_argument('--low-hz', type=float, default=30_000)
    parser.add_argument('--high-hz', type=float, default=50_000)
    parser.add_argument('--sound-speed', type=float, default=SOUND_SPEED)
    parser.add_argument('--search-range-ms', nargs=2, type=float,
                        metavar=('START', 'END'), help='Envelope correlation / peak diagnostic and waveform plot interval [ms]')
    parser.add_argument('--true-pos', nargs=2, type=float, metavar=('X', 'Y'),
                        help='Known position [m], used only for error evaluation')
    parser.add_argument('--save', type=Path, default=results_dir / 'tdoa_position.png',
                        help='Position plot path (default: results/tdoa_position.png)')
    parser.add_argument('--save-signals', type=Path, default=results_dir / 'processed_signals.png',
                        help='Waveform plot path (default: results/processed_signals.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not open plot windows')
    args = parser.parse_args()
    if not np.isfinite(args.lm_lambda) or args.lm_lambda < 0:
        parser.error('--lm-lambda must be non-negative and finite')
    if args.true_pos is not None and not np.isfinite(args.true_pos).all():
        parser.error('--true-pos requires finite coordinates')
    try:
        result = run_pipeline(args.csv, SENSORS, args.sound_speed,
                              args.low_hz, args.high_hz, args.search_range_ms,
                              solver=args.solver, lm_lambda=args.lm_lambda,
                              round_trip=None if args.propagation == 'auto' else
                              args.propagation == 'round-trip', ocp=args.ocp)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    true_pos = None
    if args.true_pos is not None:
        true_pos = np.array(args.true_pos)
    elif args.csv is None or args.csv.resolve() in {
            (DATA_DIR / 'waveforms_noisy.csv').resolve(),
            (DATA_DIR / 'waveforms_clean.csv').resolve()}:
        true_pos = EXAMPLE_TRUE_POS

    if args.csv is None:
        for label, path in zip(RX_COLUMNS, CSV_FILES):
            print(f'Input {label}: {path}')
    else:
        print(f'Input CSV: {args.csv}')
    print(f'Solver: {result["solver"]}')
    print('OCP:', 'linear_least_square' if args.solver == 'linear_least_squares' else args.ocp)
    print_peak_times(result['time_s'], result['peaks'], result['labels'], result['tdoa'])
    print('Propagation:', 'round-trip' if result['round_trip'] else 'one-way')
    print('Localization TDOA [us]:', result['localization_tdoa'] * 1e6)
    print('Distance differences [mm]:', result['localization_tdoa'] * args.sound_speed * 1000)
    print('\nEstimated position [m]:', result['position'])
    print('Estimated r0 [m]:', result['r0'])
    if true_pos is not None:
        print('True position [m] (evaluation only):', true_pos)
        print('Position error [m]:', np.linalg.norm(true_pos - result['position']))

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = []
    try:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save_signals.parent.mkdir(parents=True, exist_ok=True)
        figures.append(plot_processed(
            result['time_s'], result['raw'], result['filtered'], result['envelope'],
            result['peaks'], result['labels'], output=args.save_signals,
            show=False, time_range_ms=args.search_range_ms, close=False, tdoa=result['tdoa']))
        figures.append(visualize_tdoa(
            SENSORS, true_pos, result['position'], result['localization_tdoa'], args.sound_speed,
            output=args.save, show=False, close=False, labels=result['labels']))
        if not args.no_show:
            plt.show()  # Display both figures together.
    finally:
        for figure in figures:
            plt.close(figure)


if __name__ == '__main__':
    main()
