"""Four-channel offline ultrasonic TDOA processing.

CSV -> DC removal -> Butterworth band-pass -> Hilbert envelope ->
reference-channel cross-correlation -> three-point parabolic interpolation.
Positive TDOA means the target received later than S1. Peak times are diagnostic.

Run: python3 -m signal_process.tdoa_signal_processing --no-show
"""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import (
    butter,
    sosfiltfilt,
    hilbert,
    correlate,
    correlation_lags,
)


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = Path(__file__).resolve().parents[1] / "signal_example" / "waveforms_noisy.csv"

TIME_COLUMN = "time_s"
RX_COLUMNS = ["rx1", "rx2", "rx3", "rx4"]

LOW_HZ = 30_000.0
HIGH_HZ = 50_000.0
FILTER_ORDER = 4

SOUND_SPEED = 343.0  # m/s

SHOW_PLOTS = True


# ============================================================
# LOAD DATA
# ============================================================

def load_measurement(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Measurement file not found: {path.resolve()}"
        )

    data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig'))
    columns = data.dtype.names or ()

    sensor_columns = [f's{i}_amplitude' for i in range(1, 5)]
    if TIME_COLUMN not in columns:
        raise ValueError('CSV requires time_s')
    if all(name in columns for name in sensor_columns):
        selected = sensor_columns
        if set(name for name in columns if name.endswith('_amplitude')) != set(selected):
            raise ValueError('Expected exactly four sensor amplitude columns')
    elif all(name in columns for name in RX_COLUMNS):
        selected = RX_COLUMNS
        if any(name.startswith('rx') and name[2:].isdigit() and name not in selected
               for name in columns):
            raise ValueError('Expected exactly four receiver columns')
    else:
        raise ValueError('CSV requires s1_amplitude through s4_amplitude, or rx1 through rx4')
    time_s = np.asarray(data[TIME_COLUMN], dtype=float)
    signals = np.vstack([np.asarray(data[name], dtype=float) for name in selected])

    if signals.shape[1] != len(time_s):
        raise ValueError("Time and receiver signal lengths do not match.")

    if len(time_s) < 32:
        raise ValueError(
            f"Too few samples ({len(time_s)}). Capture a longer record."
        )

    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s contains NaN or infinite values.")

    if not np.all(np.isfinite(signals)):
        raise ValueError("Receiver data contains NaN or infinite values.")

    return time_s, signals


# ============================================================
# SAMPLING RATE
# ============================================================

def calculate_sampling_rate(time_s, high_hz=HIGH_HZ):
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or len(time_s) < 32 or not np.isfinite(time_s).all():
        raise ValueError('At least 32 finite time samples are required')
    steps = np.diff(time_s)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=0):
        raise ValueError('Time samples must be increasing and uniformly spaced')
    fs = 1 / np.mean(steps)
    if not np.isfinite(high_hz) or not 0 < high_hz < fs / 2:
        raise ValueError('Require 0 < high_hz < sample_rate / 2')
    return fs


# ============================================================
# PREPARE FILTER ONCE
# ============================================================

def make_bandpass_filter(fs, low_hz=LOW_HZ, high_hz=HIGH_HZ):
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError('Require 0 < low_hz < high_hz < sample_rate / 2')
    return butter(
        FILTER_ORDER,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


# ============================================================
# PROCESS ONE CHANNEL
#
# raw -> DC removal -> BPF -> Hilbert -> envelope
# ============================================================

def process_channel(raw_signal, sos):
    raw_signal = np.asarray(raw_signal, dtype=float)

    centered = raw_signal - np.mean(raw_signal)

    # Offline zero-phase filtering.
    filtered = sosfiltfilt(sos, centered)

    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    return filtered, envelope


# ============================================================
# PROCESS ALL CHANNELS
# ============================================================

def process_all_channels(signals, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ):
    if signals.ndim != 2 or signals.shape[0] < 2:
        raise ValueError(
            "Expected at least two receiver channels."
        )

    sos = make_bandpass_filter(fs, low_hz, high_hz)

    filtered = np.zeros_like(signals, dtype=float)
    envelopes = np.zeros_like(signals, dtype=float)

    for ch in range(signals.shape[0]):
        filtered[ch], envelopes[ch] = process_channel(
            signals[ch],
            sos,
        )

    return filtered, envelopes


# ============================================================
# SUB-SAMPLE PARABOLIC INTERPOLATION
# ============================================================

def parabolic_peak_index(y, peak_index):
    if peak_index <= 0 or peak_index >= len(y) - 1:
        return float(peak_index)

    y_minus = float(y[peak_index - 1])
    y_zero = float(y[peak_index])
    y_plus = float(y[peak_index + 1])

    denominator = y_minus - 2.0 * y_zero + y_plus

    if denominator == 0.0:
        return float(peak_index)

    delta = 0.5 * (y_minus - y_plus) / denominator

    if not np.isfinite(delta) or abs(delta) > 1.0:
        return float(peak_index)

    return float(peak_index) + delta


# ============================================================
# CROSS-CORRELATION BETWEEN RX1 AND ONE OTHER RX
#
# dT1i = Ti - T1
# positive => RXi is later than RX1
# ============================================================

def estimate_pair_tdoa(reference_envelope, target_envelope, fs):
    x_ref = (
        np.asarray(reference_envelope, dtype=float)
        - np.mean(reference_envelope)
    )
    x_target = (
        np.asarray(target_envelope, dtype=float)
        - np.mean(target_envelope)
    )

    std_ref = float(np.std(x_ref))
    std_target = float(np.std(x_target))

    eps = np.finfo(float).eps

    if std_ref <= eps or std_target <= eps:
        raise ValueError("A receiver envelope has essentially no variation.")

    # Normalize amplitude so receiver gain differences do not dominate.
    x_ref = x_ref / std_ref
    x_target = x_target / std_target

    # correlate(target, reference):
    # positive lag when target arrives later than reference.
    corr = correlate(
        x_target,
        x_ref,
        mode="full",
        method="fft",
    )

    lags = correlation_lags(
        len(x_target),
        len(x_ref),
        mode="full",
    )

    peak_index = int(np.argmax(corr))

    interpolated_index = parabolic_peak_index(
        corr,
        peak_index,
    )

    fractional_offset = interpolated_index - peak_index
    lag_samples = float(lags[peak_index]) + fractional_offset

    tdoa_s = lag_samples / fs

    return tdoa_s, lag_samples, corr, lags


# ============================================================
# RX1-REFERENCE TDOA VECTOR (one entry per non-reference channel)
#
# output:
# [dT12, dT13, dT14] for four channels
# ============================================================

def estimate_all_tdoas(envelopes, fs):
    if envelopes.ndim != 2 or envelopes.shape[0] < 2:
        raise ValueError("At least two envelopes are required.")

    reference = envelopes[0]

    tdoa_s = np.zeros(envelopes.shape[0] - 1, dtype=float)
    lag_samples = np.zeros(envelopes.shape[0] - 1, dtype=float)

    correlation_data = []

    for i in range(1, envelopes.shape[0]):
        (
            tdoa_s[i - 1],
            lag_samples[i - 1],
            corr,
            lags,
        ) = estimate_pair_tdoa(
            reference,
            envelopes[i],
            fs,
        )

        correlation_data.append((corr, lags))

    return tdoa_s, lag_samples, correlation_data


# ============================================================
# COMPLETE PROCESSING AND DIAGNOSTICS
# ============================================================

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
    fs = calculate_sampling_rate(time_s, high_hz)
    filtered, envelope = process_all_channels(raw, fs, low_hz, high_hz)
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


def process_measurement(time_s, raw, low_hz=30_000, high_hz=50_000,
                        search_range_ms=None, reference=0):
    """Return legacy peak diagnostics plus correlation TDOA in channel order.

    Filtering uses the full record; correlation uses the selected common interval.
    Peak times are diagnostic only and are never used for localization.
    """
    time_s = np.asarray(time_s, dtype=float)
    filtered, envelope, peaks = process_signals(
        time_s, raw, low_hz, high_hz, search_range_ms)
    mask = np.ones(len(time_s), dtype=bool)
    if search_range_ms is not None:
        mask = ((time_s * 1000 >= search_range_ms[0]) &
                (time_s * 1000 <= search_range_ms[1]))
    if mask.sum() < 3:
        raise ValueError('Correlation search range requires at least three samples')
    if not 0 <= reference < len(envelope):
        raise ValueError('Invalid reference channel')
    order = [reference] + [i for i in range(len(envelope)) if i != reference]
    fs = calculate_sampling_rate(time_s, high_hz)
    tdoa, lag_samples, correlation_data = estimate_all_tdoas(envelope[order][:, mask], fs)
    return dict(filtered=filtered, envelope=envelope, peaks=peaks,
                peak_times=time_s[peaks], tdoa=tdoa, lag_samples=lag_samples,
                correlation_data=correlation_data)


# ============================================================
# PLOTS
# ============================================================

def show_signal_plots(time_s, signals, filtered, envelopes,
                      low_hz=LOW_HZ, high_hz=HIGH_HZ, show=True):
    import matplotlib.pyplot as plt

    figures = []
    time_ms = time_s * 1e3

    figures.append(plt.figure())
    for i in range(len(signals)):
        plt.plot(
            time_ms,
            signals[i],
            label=f"RX{i + 1}",
        )
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.title("Raw Ultrasonic Signals")
    plt.legend()
    plt.grid(True)

    figures.append(plt.figure())
    for i in range(len(signals)):
        plt.plot(
            time_ms,
            filtered[i],
            label=f"RX{i + 1}",
        )
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.title(f"{low_hz / 1000:g}-{high_hz / 1000:g} kHz Band-pass Signals")
    plt.legend()
    plt.grid(True)

    figures.append(plt.figure())
    for i in range(len(signals)):
        plt.plot(
            time_ms,
            envelopes[i],
            label=f"RX{i + 1}",
        )
    plt.xlabel("Time [ms]")
    plt.ylabel("Envelope amplitude")
    plt.title("Hilbert Envelopes")
    plt.legend()
    plt.grid(True)

    if show:
        plt.show()
    return figures


def show_correlation_plots(correlation_data, fs, tdoa_s, show=True):
    import matplotlib.pyplot as plt

    figures = []
    for i, (corr, lags) in enumerate(correlation_data, start=2):
        lag_us = lags / fs * 1e6
        tdoa_us = tdoa_s[i - 2] * 1e6

        figures.append(plt.figure())
        plt.plot(lag_us, corr)
        plt.axvline(
            tdoa_us,
            linestyle="--",
            label=f"dT1{i} = {tdoa_us:.3f} us",
        )
        plt.xlabel("Lag [us]")
        plt.ylabel("Correlation")
        plt.title(f"Envelope Cross-Correlation: RX1 vs RX{i}")
        plt.legend()
        plt.grid(True)

    if show:
        plt.show()
    return figures


def plot_processed(time_s, raw, filtered, envelope, peaks, labels,
                   output=None, show=True, time_range_ms=None, close=True, tdoa=None, reference=0):
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
                   label=f'Envelope peak (diagnostic): {time_ms[peak]:.4f} ms')
        if tdoa is not None:
            delays = np.insert(tdoa, reference, 0.0)
            ax.text(0.02, 0.92, f'Envelope xcorr vs {labels[reference]}: {delays[i] * 1e6:+.3f} us',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
        ax.set_ylabel(f'{labels[i]}\nAmplitude [a.u.]')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    axes[0, 0].set_title('Processed ultrasound: diagnostic peaks; TDOA from envelope correlation')
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


def print_peak_times(time_s, peaks, labels, tdoa=None):
    """Print diagnostic peak times and optional correlation TDOA."""
    peak_times = time_s[peaks]
    reference = labels.index('S1') if 'S1' in labels else 0
    print(f'Sample rate: {1 / np.mean(np.diff(time_s)):.1f} Hz/channel')
    print('Envelope maximum times (not first-arrival times):')
    print('Sensor  Sample   Peak [s]      Peak [ms] (diagnostic only)')
    for label, index, peak_time in zip(labels, peaks, peak_times):
        print(f'{label:<7} {index:<8} {peak_time:.9f}   {peak_time * 1000:.6f}')
    if tdoa is not None:
        print(f'Envelope cross-correlation TDOA relative to {labels[reference]}:')
        others = [label for i, label in enumerate(labels) if i != reference]
        for label, delay in zip(others, tdoa):
            print(f'{label} - {labels[reference]}: {delay * 1e6:+.3f} us')
        print('TDOA [s] (other channels in CSV order):')
        print(tdoa)


# ============================================================
# MAIN
# ============================================================

def make_demo_measurement():
    """Generate deterministic 4-channel synthetic bursts in memory only."""
    fs = 500_000.0
    time_s = np.arange(4000) / fs
    delays = np.array([0.0, 120e-6, -80e-6, 240e-6])
    rng = np.random.default_rng(42)
    signals = []
    for delay in delays:
        local_time = time_s - (0.003 + delay)
        burst = np.exp(-0.5 * (local_time / 0.00012)**2)
        signals.append(burst * np.cos(2 * np.pi * 40_000 * local_time)
                       + rng.normal(0, 0.005, len(time_s)))
    return time_s, np.asarray(signals), delays[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--csv', type=Path, default=CSV_FILE)
    source.add_argument('--demo', action='store_true')
    parser.add_argument('--low-hz', type=float, default=LOW_HZ)
    parser.add_argument('--high-hz', type=float, default=HIGH_HZ)
    parser.add_argument('--search-range-ms', nargs=2, type=float, metavar=('START', 'END'),
                        help='Envelope correlation / peak diagnostic and plot interval [ms]')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--sound-speed', type=float, default=SOUND_SPEED)
    args = parser.parse_args()
    if not np.isfinite(args.sound_speed) or args.sound_speed <= 0:
        parser.error('--sound-speed must be positive and finite')
    try:
        if args.demo:
            time_s, raw, true_tdoa = make_demo_measurement()
        else:
            time_s, raw = load_measurement(args.csv)
        result = process_measurement(time_s, raw, args.low_hz, args.high_hz,
                                     args.search_range_ms)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    labels = [f'S{i + 1}' for i in range(len(raw))]
    print_peak_times(time_s, result['peaks'], labels, result['tdoa'])
    print('Lag [samples]:', result['lag_samples'])
    print('Distance differences [mm]:', result['tdoa'] * args.sound_speed * 1000)
    if args.demo:
        print('Demo true TDOA [us]:', true_tdoa * 1e6)
        print('Demo TDOA error [us]:', (result['tdoa'] - true_tdoa) * 1e6)
    plot_processed(time_s, raw, result['filtered'], result['envelope'], result['peaks'],
                   labels, args.save, SHOW_PLOTS and not args.no_show, args.search_range_ms,
                   tdoa=result['tdoa'])


if __name__ == '__main__':
    main()
