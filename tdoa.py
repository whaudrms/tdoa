"""Run waveform processing, peak-based TDOA localization, and plots."""

import argparse
from pathlib import Path

import numpy as np

from signal_process.process_ultrasound import (
    plot_processed, print_peak_times, process_signals,
)
from visualization.tdoa_visualization import visualize_tdoa
from visualization.waveform_visualization import DATA_DIR, load_waveforms


# Sensor coordinates [m], in S1, S2, S3, S4 order. S1 is the reference.
SENSORS = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
SOUND_SPEED = 343.0  # m/s
EXAMPLE_TRUE_POS = None  # Evaluation only; never used to estimate.


def estimate_position(sensors, tdoa, c=SOUND_SPEED):
    """Solve A @ [x, y, r1] = b using measured differences relative to S1."""
    sensors = np.asarray(sensors, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float)
    if sensors.ndim != 2 or sensors.shape[1] != 2 or len(sensors) < 4:
        raise ValueError('At least four 2D sensor coordinates are required')
    if tdoa.shape != (len(sensors) - 1,):
        raise ValueError('Provide one TDOA value per non-reference sensor')
    if not np.isfinite(sensors).all() or not np.isfinite(tdoa).all():
        raise ValueError('Sensor coordinates and TDOA must be finite')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('Sound speed must be positive and finite')
    offsets = sensors[1:] - sensors[0]
    if np.linalg.matrix_rank(offsets) < 2:
        raise ValueError('2D localization requires non-collinear sensors')
    delta_r = c * tdoa
    a = np.column_stack((offsets, delta_r))
    b = 0.5 * (np.sum(sensors[1:] ** 2, axis=1)
               - np.sum(sensors[0] ** 2) - delta_r ** 2)
    estimate = np.linalg.pinv(a) @ b
    return estimate[:2], estimate[2]


def run_pipeline(csv_path, sensors=SENSORS, c=SOUND_SPEED,
                 low_hz=30_000, high_hz=50_000, search_range_ms=None):
    """Process measured waveforms without using ground-truth positions or times."""
    time_s, raw, labels = load_waveforms(csv_path)
    expected = [f'S{i + 1}' for i in range(len(sensors))]
    if len(labels) != len(expected) or set(labels) != set(expected):
        raise ValueError(f'CSV channels must match sensor coordinates: {expected}')
    # CSV column order must not change which coordinate a signal belongs to.
    raw = raw[[labels.index(label) for label in expected]]
    filtered, envelope, peaks = process_signals(
        time_s, raw, low_hz, high_hz, search_range_ms)
    peak_times = time_s[peaks]
    tdoa = peak_times[1:] - peak_times[0]
    position, r1 = estimate_position(sensors, tdoa, c)
    return dict(time_s=time_s, raw=raw, labels=expected, filtered=filtered,
                envelope=envelope, peaks=peaks, peak_times=peak_times,
                tdoa=tdoa, position=position, r1=r1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DATA_DIR / 'waveforms_noisy.csv')
    parser.add_argument('--low-hz', type=float, default=30_000)
    parser.add_argument('--high-hz', type=float, default=50_000)
    parser.add_argument('--sound-speed', type=float, default=SOUND_SPEED)
    parser.add_argument('--search-range-ms', nargs=2, type=float,
                        metavar=('START', 'END'), help='Peak search and waveform plot interval [ms]')
    parser.add_argument('--true-pos', nargs=2, type=float, metavar=('X', 'Y'),
                        help='Known position [m], used only for error evaluation')
    parser.add_argument('--save', type=Path, help='Save the TDOA position plot')
    parser.add_argument('--save-signals', type=Path, help='Save the processed waveform plot')
    parser.add_argument('--no-show', action='store_true', help='Do not open plot windows')
    args = parser.parse_args()
    if args.true_pos is not None and not np.isfinite(args.true_pos).all():
        parser.error('--true-pos requires finite coordinates')
    try:
        result = run_pipeline(args.csv, SENSORS, args.sound_speed,
                              args.low_hz, args.high_hz, args.search_range_ms)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    true_pos = None
    if args.true_pos is not None:
        true_pos = np.array(args.true_pos)
    elif args.csv.resolve() in {
            (DATA_DIR / 'waveforms_noisy.csv').resolve(),
            (DATA_DIR / 'waveforms_clean.csv').resolve()}:
        true_pos = EXAMPLE_TRUE_POS

    print(f'Input CSV: {args.csv}')
    print_peak_times(result['time_s'], result['peaks'], result['labels'])
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
        figures.append(plot_processed(
            result['time_s'], result['raw'], result['filtered'], result['envelope'],
            result['peaks'], result['labels'], output=args.save_signals,
            show=False, time_range_ms=args.search_range_ms, close=False))
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
