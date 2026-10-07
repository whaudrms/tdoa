"""Offline ultrasonic bandpass filtering, envelope peaks, and plotting."""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from visualization.waveform_visualization import DATA_DIR, load_waveforms


def process_signals(time_s, raw, low_hz=30_000, high_hz=50_000,
                    search_range_ms=None):
    """Return filtered signals, Hilbert envelopes, and peak sample indices.

    raw has shape (channels, samples). Peaks are envelope maxima, not onsets.
    Forward/backward filtering uses future samples and is for offline analysis.
    """
    time_s = np.asarray(time_s, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if time_s.ndim != 1 or len(time_s) < 32:
        raise ValueError('At least 32 time samples are required')
    if raw.ndim != 2 or raw.shape[1] != len(time_s) or raw.shape[0] == 0:
        raise ValueError('raw must have shape (channels, samples)')
    if not np.isfinite(time_s).all() or not np.isfinite(raw).all():
        raise ValueError('Time and signals must be finite')
    steps = np.diff(time_s)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=0):
        raise ValueError('Time samples must be increasing and uniformly spaced')
    fs = 1 / np.mean(steps)
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError('Require 0 < low_hz < high_hz < sample_rate / 2')
    sos = butter(4, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos')
    centered = raw - raw.mean(axis=1, keepdims=True)
    filtered = sosfiltfilt(sos, centered, axis=1)
    envelope = np.abs(hilbert(filtered, axis=1))
    mask = np.ones(len(time_s), dtype=bool)
    if search_range_ms is not None:
        limits = np.asarray(search_range_ms, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError('Search range requires finite START < END in ms')
        mask = (time_s * 1000 >= limits[0]) & (time_s * 1000 <= limits[1])
        if not mask.any():
            raise ValueError('Search range contains no samples')
    candidates = np.flatnonzero(mask)
    peaks = candidates[np.argmax(envelope[:, mask], axis=1)]
    return filtered, envelope, peaks


def plot_processed(time_s, raw, filtered, envelope, peaks, labels,
                   output=None, show=True, time_range_ms=None, close=True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(labels), 1, sharex=True, sharey=True,
                             figsize=(12, 2.2 * len(labels)), squeeze=False,
                             constrained_layout=True)
    time_ms = time_s * 1000
    for i, ax in enumerate(axes[:, 0]):
        peak = peaks[i]
        ax.plot(time_ms, raw[i], color='gray', alpha=0.35, linewidth=0.6, label='Raw')
        ax.plot(time_ms, filtered[i], color='tab:blue', linewidth=0.8, label='Bandpass')
        ax.plot(time_ms, envelope[i], color='tab:orange', linewidth=1.4, label='Envelope')
        ax.scatter(time_ms[peak], envelope[i, peak], color='red', zorder=5)
        ax.axvline(time_ms[peak], color='red', linestyle='--', linewidth=1,
                   label=f'Envelope peak: {time_ms[peak]:.4f} ms')
        ax.set_ylabel(f'{labels[i]}\nAmplitude [a.u.]')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    axes[0, 0].set_title('Processed ultrasound: envelope peaks (not arrival onsets)')
    axes[-1, 0].set_xlabel('Time since recording start [ms]')
    if time_range_ms is not None:
        axes[-1, 0].set_xlim(*time_range_ms)
    if output:
        fig.savefig(output, dpi=160)
        print(f'Processed plot saved to: {output}')
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


def print_peak_times(time_s, peaks, labels):
    """Print measured envelope peak times and differences in channel order."""
    peak_times = time_s[peaks]
    reference = labels.index('S1') if 'S1' in labels else 0
    print(f'Sample rate: {1 / np.mean(np.diff(time_s)):.1f} Hz/channel')
    print('Envelope maximum times (not first-arrival times):')
    print(f'Sensor  Sample   Peak [s]      Peak [ms]    Difference to {labels[reference]} [us]')
    for label, index, peak_time in zip(labels, peaks, peak_times):
        difference_us = (peak_time - peak_times[reference]) * 1e6
        print(f'{label:<7} {index:<8} {peak_time:.9f}   {peak_time * 1000:.6f}   {difference_us:+.3f}')
    print('Peak-based TDOA [s] (other channels in CSV order):')
    print(np.delete(peak_times - peak_times[reference], reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DATA_DIR / 'waveforms_noisy.csv')
    parser.add_argument('--low-hz', type=float, default=30_000)
    parser.add_argument('--high-hz', type=float, default=50_000)
    parser.add_argument('--search-range-ms', nargs=2, type=float,
                        metavar=('START', 'END'), help='Peak search and plot interval [ms]')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    time_s, raw, labels = load_waveforms(args.csv)
    filtered, envelope, peaks = process_signals(
        time_s, raw, args.low_hz, args.high_hz, args.search_range_ms)
    print_peak_times(time_s, peaks, labels)
    plot_processed(time_s, raw, filtered, envelope, peaks, labels,
                   args.save, not args.no_show, args.search_range_ms)


if __name__ == '__main__':
    main()
