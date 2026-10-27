"""
5-Channel Airborne Ultrasonic TDOA Signal Processing
====================================================

Input:
    A single-burst, time-aligned CSV acquired simultaneously from 5 receivers.

Required CSV columns:
    time_s,rx1,rx2,rx3,rx4,rx5

Signal-processing chain:
    raw waveform
      -> DC removal
      -> 34-46 kHz Butterworth band-pass
      -> Hilbert analytic signal
      -> envelope
      -> envelope cross-correlation against RX1
      -> 3-point parabolic peak interpolation
      -> [dT12, dT13, dT14, dT15]

Sign convention:
    dT1i = Ti - T1

Therefore:
    positive dT12 => RX2 received later than RX1

Important:
    - All five channels must share the same acquisition clock/time base.
    - This file expects raw ultrasonic waveforms, not comparator edges.
    - This stage estimates TDOAs only; no position solver is connected.

Standalone usage:
    python3 tdoa_signal_processing_5ch.py --csv measurement_5ch.csv
    python3 tdoa_signal_processing_5ch.py --demo
    python3 tdoa_signal_processing_5ch.py --demo --no-show

Dependencies: numpy, scipy, matplotlib (for plots only).
"""

import argparse
from pathlib import Path

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

CSV_FILE = Path(__file__).resolve().parent / "measurement_5ch.csv"

TIME_COLUMN = "time_s"
RX_COLUMNS = ["rx1", "rx2", "rx3", "rx4", "rx5"]

LOW_HZ = 34_000.0
HIGH_HZ = 46_000.0
FILTER_ORDER = 4

SOUND_SPEED = 344.4  # m/s

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

    required = [TIME_COLUMN] + RX_COLUMNS
    missing = [name for name in required if name not in columns]

    if missing:
        raise ValueError(
            "Required CSV column(s) missing: "
            + ", ".join(missing)
            + f"\nCurrent columns: {list(columns)}"
        )

    time_s = np.asarray(data[TIME_COLUMN], dtype=float)

    signals = np.vstack(
        [np.asarray(data[name], dtype=float) for name in RX_COLUMNS]
    )

    if signals.shape[1] != len(time_s):
        raise ValueError("Time and receiver signal lengths do not match.")

    if len(time_s) < 64:
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

def calculate_sampling_rate(time_s):
    dt = np.diff(time_s)

    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing.")

    sample_period = float(np.median(dt))
    fs = 1.0 / sample_period

    # TDOA estimation assumes one common, nearly uniform sampling grid.
    relative_jitter = float(
        np.max(np.abs(dt - sample_period)) / sample_period
    )

    if relative_jitter > 0.01:
        raise ValueError(
            "Sampling interval is not sufficiently uniform. "
            f"Maximum relative deviation = {relative_jitter * 100:.2f}%"
        )

    if fs <= 2.0 * HIGH_HZ:
        raise ValueError(
            "Sampling rate is too low for the selected band-pass filter.\n"
            f"fs = {fs:.1f} Hz, Nyquist = {fs / 2:.1f} Hz, "
            f"HIGH_HZ = {HIGH_HZ:.1f} Hz"
        )

    return fs


# ============================================================
# PREPARE FILTER ONCE
# ============================================================

def make_bandpass_filter(fs):
    return butter(
        FILTER_ORDER,
        [LOW_HZ, HIGH_HZ],
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
# PROCESS ALL 5 CHANNELS
# ============================================================

def process_all_channels(signals, fs):
    if signals.shape[0] != 5:
        raise ValueError(
            f"Expected 5 receiver channels, got {signals.shape[0]}."
        )

    sos = make_bandpass_filter(fs)

    filtered = np.zeros_like(signals, dtype=float)
    envelopes = np.zeros_like(signals, dtype=float)

    for ch in range(5):
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
# RX1-REFERENCE TDOA VECTOR
#
# output:
# [dT12, dT13, dT14, dT15]
# ============================================================

def estimate_all_tdoas(envelopes, fs):
    if envelopes.shape[0] != 5:
        raise ValueError("Five envelopes are required.")

    reference = envelopes[0]

    tdoa_s = np.zeros(4, dtype=float)
    lag_samples = np.zeros(4, dtype=float)

    correlation_data = []

    for i in range(1, 5):
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
# PLOTS
# ============================================================

def show_signal_plots(time_s, signals, filtered, envelopes):
    import matplotlib.pyplot as plt

    time_ms = time_s * 1e3

    plt.figure()
    for i in range(5):
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

    plt.figure()
    for i in range(5):
        plt.plot(
            time_ms,
            filtered[i],
            label=f"RX{i + 1}",
        )
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.title("34-46 kHz Band-pass Signals")
    plt.legend()
    plt.grid(True)

    plt.figure()
    for i in range(5):
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


def show_correlation_plots(correlation_data, fs, tdoa_s):
    import matplotlib.pyplot as plt

    for i, (corr, lags) in enumerate(correlation_data, start=2):
        lag_us = lags / fs * 1e6
        tdoa_us = tdoa_s[i - 2] * 1e6

        plt.figure()
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

    plt.show()


# ============================================================
# MAIN
# ============================================================

def make_demo_measurement():
    """Generate deterministic 5-channel synthetic bursts in memory only."""
    fs = 500_000.0
    time_s = np.arange(4000) / fs
    delays = np.array([0.0, 120e-6, -80e-6, 240e-6, 50e-6])
    rng = np.random.default_rng(42)
    signals = []
    for delay in delays:
        local_time = time_s - (0.003 + delay)
        burst = np.exp(-0.5 * (local_time / 0.00012)**2)
        signals.append(burst * np.cos(2 * np.pi * 40_000 * local_time)
                       + rng.normal(0, 0.005, len(time_s)))
    return time_s, np.asarray(signals), delays[1:]


def main():
    parser = argparse.ArgumentParser(
        description='Standalone 5-channel envelope cross-correlation TDOA estimation',
        epilog='CSV columns: time_s,rx1,rx2,rx3,rx4,rx5. No position solver is used.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--csv', type=Path, help='Measurement CSV (default: measurement_5ch.csv beside this script)')
    source.add_argument('--demo', action='store_true', help='Run synthetic 5-channel data without a CSV')
    parser.add_argument('--no-show', action='store_true', help='Print results without opening plots')
    parser.add_argument('--sound-speed', type=float, default=SOUND_SPEED,
                        help='Sound speed for distance differences [m/s]')
    args = parser.parse_args()
    if not np.isfinite(args.sound_speed) or args.sound_speed <= 0:
        parser.error('--sound-speed must be positive and finite')
    csv_path = args.csv if args.csv is not None else CSV_FILE
    try:
        if args.demo:
            time_s, signals, true_tdoa = make_demo_measurement()
        else:
            time_s, signals = load_measurement(csv_path)
        fs = calculate_sampling_rate(time_s)
        filtered, envelopes = process_all_channels(signals, fs)
        tdoa_s, lag_samples, correlation_data = estimate_all_tdoas(envelopes, fs)
    except (ValueError, OSError) as exc:
        parser.error(f'{exc}\nUse --csv PATH with columns time_s,rx1,rx2,rx3,rx4,rx5, '
                     'or --demo for synthetic data.')

    tdoa_us = tdoa_s * 1e6
    path_difference_mm = args.sound_speed * tdoa_s * 1000.0

    print("=" * 68)
    print("5-Channel Ultrasonic TDOA Signal Processing")
    print("=" * 68)
    print(f"Input         : {'synthetic demo (not measured data)' if args.demo else csv_path}")
    print(f"Samples       : {len(time_s)}")
    print(f"Sampling rate : {fs / 1000.0:.3f} kS/s")
    print()

    for i in range(4):
        receiver_number = i + 2

        print(
            f"dT1{receiver_number} = "
            f"{tdoa_us[i]:+10.3f} us   |   "
            f"lag = {lag_samples[i]:+9.4f} samples   |   "
            f"delta_d = {path_difference_mm[i]:+10.3f} mm"
        )

    print()
    print("TDOA vector [s]:")
    print(tdoa_s)
    print("=" * 68)

    if args.demo:
        print('Demo true TDOA [us]:', true_tdoa * 1e6)
        print('Demo TDOA error [us]:', (tdoa_s - true_tdoa) * 1e6)

    if SHOW_PLOTS and not args.no_show:
        show_signal_plots(
            time_s,
            signals,
            filtered,
            envelopes,
        )

        show_correlation_plots(
            correlation_data,
            fs,
            tdoa_s,
        )


if __name__ == "__main__":
    main()
