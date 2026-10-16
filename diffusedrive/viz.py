"""Visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_scene(samples: np.ndarray, gt: np.ndarray, map_polylines: np.ndarray,
               save_path: Path, title: str = "Scene"):
    """Plot K sampled futures vs ground truth. samples: [K, A, T, 2]"""
    K, A, T, _ = samples.shape

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot map
    for p in range(map_polylines.shape[0]):
        poly = map_polylines[p]
        valid = ~np.isnan(poly[:, 0])
        if valid.any():
            ax.plot(poly[valid, 0], poly[valid, 1], 'k-', alpha=0.3, linewidth=1)

    # Plot ground truth
    colors = plt.cm.tab10(np.arange(A))
    for a in range(A):
        hist_pos = gt[a, :, :2]
        fut_pos = gt[a, -1:, :2]
        ax.plot(hist_pos[:, 0], hist_pos[:, 1], 'o-', color=colors[a], markersize=4, label=f"Agent {a}")
        ax.plot(fut_pos[:, 0], fut_pos[:, 1], 's', color=colors[a], markersize=8, markeredgecolor='red', linewidth=2)

    # Plot samples
    for k in range(min(K, 6)):
        for a in range(A):
            alpha = 0.3 + 0.4 * (k / K)
            ax.plot(samples[k, a, :, 0], samples[k, a, :, 1], '--', color=colors[a], alpha=alpha, linewidth=1)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
