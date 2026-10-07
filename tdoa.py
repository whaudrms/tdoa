import numpy as np
import argparse


def visualize_tdoa(sensors, true_pos, estimated_pos, tdoa, c, output=None,
                   show=True):
    """Plot sensor geometry, measured TDOA contours, and localization error."""
    import matplotlib.pyplot as plt

    fig, (ax, time_ax) = plt.subplots(1, 2, figsize=(12, 5),
                                    constrained_layout=True)
    points = np.vstack((sensors, true_pos, estimated_pos))
    lower, upper = points.min(axis=0), points.max(axis=0)
    margin = max(np.max(upper - lower) * 0.35, 0.25)
    lower, upper = lower - margin, upper + margin
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 500),
                         np.linspace(lower[1], upper[1], 500))
    reference_distance = np.hypot(xx - sensors[0, 0], yy - sensors[0, 1])
    colors = plt.get_cmap('tab10').colors
    for i, delay in enumerate(tdoa, start=1):
        difference = np.hypot(xx - sensors[i, 0], yy - sensors[i, 1])
        difference -= reference_distance
        level = c * delay
        color = colors[(i - 1) % len(colors)]
        if difference.min() < level < difference.max():
            ax.contour(xx, yy, difference, levels=[level], colors=[color],
                       linewidths=1.5)
            ax.plot([], [], color=color, label=f'S{i + 1} - S1 TDOA')

    ax.scatter(sensors[:, 0], sensors[:, 1], marker='^', s=90,
               color='black', label='Sensors', zorder=4)
    for i, (x, y) in enumerate(sensors, start=1):
        ax.annotate(f'S{i}' + (' (reference)' if i == 1 else ''),
                    (x, y), xytext=(6, 8), textcoords='offset points')
    ax.scatter(*true_pos, marker='*', s=180, color='green',
               label='True source', zorder=5)
    ax.scatter(*estimated_pos, marker='x', s=100, linewidths=2,
               color='red', label='Estimated source', zorder=6)
    ax.plot([true_pos[0], estimated_pos[0]],
            [true_pos[1], estimated_pos[1]], '--', color='red', alpha=0.6)
    error = np.linalg.norm(true_pos - estimated_pos)
    ax.set(title=f'TDOA localization (error: {error:.6f} m)',
           xlabel='X [m]', ylabel='Y [m]',
           xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    labels = [f'S{i + 1} - S1' for i in range(1, len(sensors))]
    delays_ms = tdoa * 1000
    time_ax.bar(labels, delays_ms,
                color=[colors[i % len(colors)] for i in range(len(tdoa))])
    time_ax.axhline(0, color='black', linewidth=0.8)
    time_ax.set(title='Arrival time difference relative to S1',
                xlabel='Sensor pair', ylabel='TDOA [ms]')
    time_ax.grid(axis='y', alpha=0.25)
    time_ax.margins(y=0.2)
    for i, delay in enumerate(delays_ms):
        time_ax.annotate(f'{delay:.4f}', (i, delay),
                         xytext=(0, 5 if delay >= 0 else -5),
                         textcoords='offset points', ha='center',
                         va='bottom' if delay >= 0 else 'top')
    if output:
        fig.savefig(output, dpi=160)
        print(f'\nVisualization saved to: {output}')
    if show:
        plt.show()
    plt.close(fig)

# --------------------------------------------------
# 1. Sensor positions [m]
# --------------------------------------------------
sensors = np.array([
    [0.0, 0.0],   # sensor 1: reference
    [1.0, 0.0],   # sensor 2
    [0.0, 1.0],   # sensor 3
    [1.0, 1.0],   # sensor 4
])

# Speed of sound [m/s]
c = 343.0


# --------------------------------------------------
# 2. Example: generate TDOA measurements
# --------------------------------------------------
true_pos = np.array([0.4, 0.6])

# Distance from source to each sensor
distances = np.linalg.norm(sensors - true_pos, axis=1)

# Time of arrival
toa = distances / c

# Reference sensor = sensor 1
tdoa = toa[1:] - toa[0]

print("TDOA [s]:")
print(tdoa)


# --------------------------------------------------
# 3. Convert TDOA to distance difference
# --------------------------------------------------
delta_r = c * tdoa

# delta_r[i] = r_(i+2) - r_1


# --------------------------------------------------
# 4. Construct A z = b
#
# z = [x, y, r1]^T
# --------------------------------------------------
x1, y1 = sensors[0]

A = []
b = []

for i in range(1, len(sensors)):
    xi, yi = sensors[i]
    dr = delta_r[i - 1]

    A.append([
        xi - x1,
        yi - y1,
        dr
    ])

    bi = 0.5 * (
        xi**2 + yi**2
        - x1**2 - y1**2
        - dr**2
    )

    b.append(bi)

A = np.array(A)
b = np.array(b)


# --------------------------------------------------
# 5. Solve using pseudo-inverse
# --------------------------------------------------
z_hat = np.linalg.pinv(A) @ b

x_hat = z_hat[0]
y_hat = z_hat[1]
r1_hat = z_hat[2]


# --------------------------------------------------
# 6. Result
# --------------------------------------------------
print("\nTrue position:")
print(true_pos)

print("\nEstimated position:")
print([x_hat, y_hat])

print("\nEstimated r1:")
print(r1_hat)

error = np.linalg.norm(true_pos - np.array([x_hat, y_hat]))
print("\nPosition error [m]:")
print(error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDOA localization visualization')
    parser.add_argument('--save', metavar='PATH', help='Save the figure to a file')
    parser.add_argument('--no-show', action='store_true',
                        help='Run without opening a plot window')
    args = parser.parse_args()
    visualize_tdoa(sensors, true_pos, np.array([x_hat, y_hat]), tdoa, c,
                   output=args.save, show=not args.no_show)
