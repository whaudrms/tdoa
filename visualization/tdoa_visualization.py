"""TDOA localization plotting utilities."""

import numpy as np


def visualize_tdoa(sensors, true_pos, estimated_pos, tdoa, c, output=None,
                   show=True, close=True):
    """Plot sensor geometry, measured TDOA contours, and localization error."""
    import matplotlib.pyplot as plt

    fig, (ax, time_ax) = plt.subplots(1, 2, figsize=(12, 5),
                                    constrained_layout=True)
    points = np.vstack((sensors, estimated_pos))
    if true_pos is not None:
        points = np.vstack((points, true_pos))
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
                       linewidths=1.5, linestyles='solid')
            ax.plot([], [], color=color, label=f'S{i + 1} - S1 TDOA')

    ax.scatter(sensors[:, 0], sensors[:, 1], marker='^', s=90,
               color='black', label='Sensors', zorder=4)
    for i, (x, y) in enumerate(sensors, start=1):
        ax.annotate(f'S{i}' + (' (reference)' if i == 1 else ''),
                    (x, y), xytext=(6, 8), textcoords='offset points')
    title = 'TDOA localization'
    if true_pos is not None:
        ax.scatter(*true_pos, marker='*', s=180, color='green',
                   label='True source', zorder=5)
        ax.plot([true_pos[0], estimated_pos[0]],
                [true_pos[1], estimated_pos[1]], '--', color='red', alpha=0.6)
        error = np.linalg.norm(true_pos - estimated_pos)
        title += f' (error: {error:.6f} m)'
    ax.scatter(*estimated_pos, marker='x', s=100, linewidths=2,
               color='red', label='Estimated position', zorder=6)
    ax.set(title=title,
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
    if close:
        plt.close(fig)
    return fig
