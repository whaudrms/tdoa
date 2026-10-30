"""Shared sensor selection for tdoa.py, run.py and signal processing.

Edit SELECTED_SENSORS below. Each entry is (sensor name, sample number).
The first entry is the reference. Select at least three distinct sensors.
Coordinates are in meters; CSV indices are ignored and file order is retained.
"""

from pathlib import Path

SAMPLE_ROOT = Path(__file__).resolve().parent / 'real_signal' / 'real_sample_261030'
SAMPLE_RATE_HZ = 500_000.0  # 2 us

# Coordinate label in cm: (folder, (x_m, y_m)).
SENSOR_CATALOG = {
    '0_20': ('0_20', (0.0, 0.20)),
    '0_15': ('0_15', (0.0, 0.15)),
    '20_0': ('20_0', (0.20, 0.0)),
    '0_30': ('0_30', (0.0, 0.30)),
    '20_40': ('20_40', (0.20, 0.40)),
}

# Change names, order, or sample numbers here. Example: ('20_40', 1). # rx0 , rx1 , rx2, rx3
SELECTED_SENSORS = [
    ('0_20', 1),
    ('20_0', 1),
    ('20_40', 1),
    ('0_30', 1)
]


def selected_sensor_data():
    names = [name for name, _ in SELECTED_SENSORS]
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError('Select at least three distinct sensors in sensor_config.py')
    paths, coordinates = [], []
    for name, sample in SELECTED_SENSORS:
        if name not in SENSOR_CATALOG:
            raise ValueError(f'Unknown sensor: {name}')
        if not isinstance(sample, int) or isinstance(sample, bool) or sample < 1:
            raise ValueError(f'{name}: sample number must be a positive integer')
        folder, position = SENSOR_CATALOG[name]
        paths.append(SAMPLE_ROOT / folder / f'{folder}_sample_{sample}.csv')
        coordinates.append(position)
    return names, paths, coordinates
