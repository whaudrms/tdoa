"""Run waveform processing, envelope cross-correlation TDOA localization, and plots."""

import argparse
from pathlib import Path

import numpy as np

from optimization.linear_least_squares import SOUND_SPEED, estimate_position
from optimization.gradient_descent import estimate_position as estimate_gradient_descent
from optimization.newton_method import estimate_position as estimate_newton
from optimization.gauss_newton import estimate_position as estimate_gauss_newton
from optimization.levenberg_marquardt import estimate_position as estimate_lm
from signal_process.tdoa_signal_processing import (
    plot_processed, print_peak_times, process_measurement, load_measurement,
)
from visualization.tdoa_visualization import visualize_tdoa
from visualization.waveform_visualization import DATA_DIR


# Sensor coordinates [m], in S1, S2, S3, S4 order. S1 is the reference.
SENSORS = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
SOLVERS = {'linear_least_squares': estimate_position,
           'gradient_descent': estimate_gradient_descent,
           'newton_method': estimate_newton,
           'gauss_newton': estimate_gauss_newton,
           'levenberg_marquardt': estimate_lm}
EXAMPLE_TRUE_POS = None  # Evaluation only; never used to estimate.


def run_pipeline(csv_path, sensors=SENSORS, c=SOUND_SPEED,
                 low_hz=30_000, high_hz=50_000, search_range_ms=None,
                 solver='linear_least_squares', lm_lambda=1e-3):
    """Process measured waveforms without using ground-truth positions or times."""
    if solver not in SOLVERS:
        raise ValueError(f'Unknown solver: {solver}; choose from {tuple(SOLVERS)}')
    time_s, raw = load_measurement(csv_path)
    expected = [f'S{i + 1}' for i in range(len(sensors))]
    if len(raw) != len(sensors):
        raise ValueError('CSV channels must match sensor coordinates')
    processed = process_measurement(time_s, raw, low_hz, high_hz, search_range_ms)
    tdoa = processed['tdoa']
    solver_options = {'damping': lm_lambda} if solver == 'levenberg_marquardt' else {}
    position, r1 = SOLVERS[solver](sensors, tdoa, c, **solver_options)
    return dict(time_s=time_s, raw=raw, labels=expected, **processed,
                position=position, r1=r1, solver=solver)


def main():
    results_dir = Path(__file__).resolve().parent / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DATA_DIR / 'waveforms_noisy.csv')
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
                              solver=args.solver, lm_lambda=args.lm_lambda)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    true_pos = None
    if args.true_pos is not None:
        true_pos = np.array(args.true_pos)
    elif args.csv.resolve() in {
            (DATA_DIR / 'waveforms_noisy.csv').resolve(),
            (DATA_DIR / 'waveforms_clean.csv').resolve()}:
        true_pos = EXAMPLE_TRUE_POS

    print(f'Input CSV: {args.csv}')
    print(f'Solver: {result["solver"]}')
    print_peak_times(result['time_s'], result['peaks'], result['labels'], result['tdoa'])
    print('\nEstimated position [m]:', result['position'])
    print('Estimated r1 [m]:', result['r1'])
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
            SENSORS, true_pos, result['position'], result['tdoa'], args.sound_speed,
            output=args.save, show=False, close=False))
        if not args.no_show:
            plt.show()  # Display both figures together.
    finally:
        for figure in figures:
            plt.close(figure)


if __name__ == '__main__':
    main()
