"""Plot multichannel waveform CSVs and optional reference arrival times."""

import argparse
import csv
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / 'signal_example'


def load_waveforms(path):
    """Return time [s], raw (channels, samples), and channel labels."""
    data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
    names = data.dtype.names or ()
    columns = [name for name in names if name.endswith('_amplitude')]
    if 'time_s' not in names or not columns:
        raise ValueError('CSV requires time_s and sensor *_amplitude columns')
    raw = np.vstack([data[name] for name in columns])
    labels = [name.removesuffix('_amplitude').upper() for name in columns]
    return data['time_s'], raw, labels


def load_arrival_times(path, labels):
    """Match reference arrival times to channels by sensor_id, not row order."""
    with Path(path).open(newline='', encoding='utf-8') as handle:
        arrivals = {}
        for row in csv.DictReader(handle):
            sensor = row['sensor_id'].upper()
            if sensor in arrivals:
                raise ValueError(f'Duplicate sensor_id: {sensor}')
            arrivals[sensor] = float(row['arrival_time_s'])
    return np.array([arrivals[label] for label in labels])


def visualize_waveforms(time_s, raw, labels=None, arrival_times=None,
                        output=None, show=True, time_range_ms=None):
    """Plot one panel per channel; all input times are in seconds."""
    import matplotlib.pyplot as plt

    time_s = np.asarray(time_s, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError('time_s must contain at least two samples')
    if raw.ndim != 2 or raw.shape[1] != time_s.size or raw.shape[0] == 0:
        raise ValueError('raw must have shape (channels, samples)')
    if not np.isfinite(time_s).all() or not np.isfinite(raw).all():
        raise ValueError('Waveforms and time must be finite')
    if not np.all(np.diff(time_s) > 0):
        raise ValueError('time_s must be strictly increasing')
    labels = list(labels) if labels is not None else [
        f'S{i + 1}' for i in range(raw.shape[0])]
    if len(labels) != raw.shape[0]:
        raise ValueError('One label is required per channel')
    if arrival_times is not None:
        arrival_times = np.asarray(arrival_times, dtype=float)
        if arrival_times.shape != (raw.shape[0],) or not np.isfinite(arrival_times).all():
            raise ValueError('One finite arrival time is required per channel')
    if time_range_ms is not None:
        limits = np.asarray(time_range_ms, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError('Time range requires finite START < END in ms')

    fig, axes = plt.subplots(raw.shape[0], 1, sharex=True, sharey=True,
                             figsize=(12, 2 * raw.shape[0]), squeeze=False,
                             constrained_layout=True)
    colors = plt.get_cmap('tab10').colors
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(time_s * 1000, raw[i], color=colors[i % len(colors)],
                linewidth=0.8, label=labels[i])
        if arrival_times is not None:
            arrival_ms = arrival_times[i] * 1000
            ax.axvline(arrival_ms, color='red', linestyle='--', linewidth=1,
                       label=f'True arrival: {arrival_ms:.4f} ms')
        ax.set_ylabel(f'{labels[i]}\nAmplitude [a.u.]')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=8)
    axes[0, 0].set_title('Ultrasonic sensor waveforms')
    axes[-1, 0].set_xlabel('Time since recording start [ms]')
    if time_range_ms is not None:
        axes[-1, 0].set_xlim(*limits)
    if output:
        fig.savefig(output, dpi=160)
        print(f'Waveform plot saved to: {output}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DATA_DIR / 'waveforms_noisy.csv',
                        help='Waveform CSV (default: synthetic noisy recording)')
    parser.add_argument('--truth', type=Path,
                        help='Optional ground_truth.csv with reference arrival times')
    parser.add_argument('--save', type=Path, help='Save the plot to a file')
    parser.add_argument('--no-show', action='store_true', help='Do not open a window')
    parser.add_argument('--time-range-ms', nargs=2, type=float, metavar=('START', 'END'),
                        help='Zoom the time axis to START END in milliseconds')
    args = parser.parse_args()
    time_s, raw, labels = load_waveforms(args.csv)
    arrivals = load_arrival_times(args.truth, labels) if args.truth else None
    visualize_waveforms(time_s, raw, labels, arrivals, args.save,
                        not args.no_show, args.time_range_ms)


if __name__ == '__main__':
    main()
