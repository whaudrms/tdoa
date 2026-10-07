"""Generate reproducible, synchronized synthetic ultrasonic recordings."""

import csv
from pathlib import Path

import numpy as np


# Simulation assumptions; these are not measured hardware specifications.
SENSORS = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
SOURCE = np.array([0.4, 0.6])
SOUND_SPEED = 343.0
SAMPLE_RATE = 500_000.0  # Hz per channel, simultaneous sampling
CARRIER = 40_000.0  # Hz
BURST_CYCLES = 12
EMISSION_TIME = 0.001  # s after recording starts
RECORD_DURATION = 0.008  # s, exclusive end
AMPLITUDES = np.array([1.0, 0.8, 0.9, 0.7])
NOISE_STD = 0.03  # arbitrary amplitude units
SEED = 42


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate(output_dir=None):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    count = int(round(RECORD_DURATION * SAMPLE_RATE))
    time = np.arange(count) / SAMPLE_RATE
    distances = np.linalg.norm(SENSORS - SOURCE, axis=1)
    propagation = distances / SOUND_SPEED
    arrivals = EMISSION_TIME + propagation
    duration = BURST_CYCLES / CARRIER
    clean = np.zeros((count, len(SENSORS)))

    for channel, arrival in enumerate(arrivals):
        relative_time = time - arrival
        active = (relative_time >= 0) & (relative_time < duration)
        local = relative_time[active]
        # Continuous-time Hann envelope and carrier evaluated at sample times.
        # Arrival times remain fractional samples rather than rounded shifts.
        envelope = np.sin(np.pi * local / duration) ** 2
        clean[active, channel] = (
            AMPLITUDES[channel] * envelope * np.sin(2 * np.pi * CARRIER * local)
        )

    rng = np.random.default_rng(SEED)
    noisy = clean + rng.normal(0, NOISE_STD, clean.shape)
    header = ['sample_index', 'time_s', 's1_amplitude', 's2_amplitude',
              's3_amplitude', 's4_amplitude']
    for name, waveform in [('waveforms_clean.csv', clean),
                           ('waveforms_noisy.csv', noisy)]:
        write_csv(output_dir / name, header,
                  ((i, time[i], *waveform[i]) for i in range(count)))

    write_csv(output_dir / 'ground_truth.csv',
              ['sensor_id', 'sensor_x_m', 'sensor_y_m', 'source_x_m',
               'source_y_m', 'distance_m', 'emission_time_s',
               'propagation_time_s', 'arrival_time_s', 'tdoa_to_s1_s',
               'arrival_sample_fractional', 'first_sample_at_or_after_arrival',
               'amplitude'],
              ((f'S{i + 1}', *SENSORS[i], *SOURCE, distances[i],
                EMISSION_TIME, propagation[i], arrivals[i],
                arrivals[i] - arrivals[0], arrivals[i] * SAMPLE_RATE,
                int(np.ceil(arrivals[i] * SAMPLE_RATE)), AMPLITUDES[i])
               for i in range(len(SENSORS))))
    parameters = [
        ('sample_rate_hz_per_channel', SAMPLE_RATE, 'Hz'),
        ('carrier_frequency_hz', CARRIER, 'Hz'),
        ('burst_cycles', BURST_CYCLES, 'cycles'),
        ('burst_duration_s', duration, 's'),
        ('emission_time_s', EMISSION_TIME, 's'),
        ('record_duration_s', RECORD_DURATION, 's'),
        ('sample_count_per_channel', count, 'samples'),
        ('sound_speed_m_per_s', SOUND_SPEED, 'm/s'),
        ('noise_std', NOISE_STD, 'arbitrary_amplitude_units'),
        ('random_seed', SEED, 'integer'),
        ('channel_time_offset_s', 0, 's'),
        ('reflection_amplitude', 0, 'arbitrary_amplitude_units'),
        ('dc_offset', 0, 'arbitrary_amplitude_units'),
    ]
    write_csv(output_dir / 'parameters.csv', ['parameter', 'value', 'unit'],
              parameters)
    print(f'Generated 4 CSV files in {output_dir} ({count} samples/channel)')


if __name__ == '__main__':
    generate()
