"""Forecasting metrics."""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple


def minADE_k(pred: np.ndarray, gt: np.ndarray, k: int = 6) -> float:
    """Minimum Average Displacement Error over K samples. pred: [K, A, T, 2], gt: [A, T, 2]"""
    pred = pred[:k]
    ade = np.linalg.norm(pred - gt[None], axis=(2, 3)).mean(axis=2)  # [K, A]
    return ade.min(axis=0).mean()


def minFDE_k(pred: np.ndarray, gt: np.ndarray, k: int = 6) -> float:
    """Minimum Final Displacement Error over K samples."""
    pred = pred[:k]
    fde = np.linalg.norm(pred[:, :, -1] - gt[None, :, -1], axis=2)  # [K, A]
    return fde.min(axis=0).mean()


def miss_rate(pred: np.ndarray, gt: np.ndarray, threshold: float = 2.0, k: int = 6) -> float:
    """% of agents with minFDE > threshold."""
    pred = pred[:k]
    fde = np.linalg.norm(pred[:, :, -1] - gt[None, :, -1], axis=2)
    best_fde = fde.min(axis=0)
    return (best_fde > threshold).mean()


def jointADE(pred: np.ndarray, gt: np.ndarray, k: int = 6) -> float:
    """Joint ADE (WOMD-style): mean of per-timestep min ADE over K."""
    pred = pred[:k]
    ade_t = np.linalg.norm(pred - gt[None], axis=(1, 3)).min(axis=0)  # [T]
    return ade_t.mean()


def collision_rate(pred: np.ndarray, radius: float = 0.5) -> float:
    """% of timesteps with pairwise overlap. pred: [K, A, T, 2]"""
    K, A, T, _ = pred.shape
    collisions = 0
    total = K * T

    for k in range(K):
        for t in range(T):
            pos = pred[k, :, t]  # [A, 2]
            dist = cdist(pos, pos)
            np.fill_diagonal(dist, np.inf)
            collisions += (dist < 2 * radius).any()

    return collisions / total if total > 0 else 0.0


def nll_kde(pred: np.ndarray, gt: np.ndarray, bandwidth: float = 0.5) -> float:
    """Negative log-likelihood under KDE of K samples. pred: [K, A, T, 2], gt: [A, T, 2]"""
    from scipy.stats import gaussian_kde

    pred_flat = pred.reshape(-1, 2)  # [K*A*T, 2]
    gt_flat = gt.reshape(-1, 2)  # [A*T, 2]

    try:
        kde = gaussian_kde(pred_flat.T, bw_method=bandwidth)
        nll = -np.log(kde(gt_flat.T) + 1e-6).mean()
        return nll
    except:
        return np.inf


def compute_metrics(pred: np.ndarray, gt: np.ndarray, K: int = 6) -> dict:
    """Compute all metrics."""
    return {
        "minADE_k": minADE_k(pred, gt, K),
        "minFDE_k": minFDE_k(pred, gt, K),
        "miss_rate": miss_rate(pred, gt, k=K),
        "jointADE": jointADE(pred, gt, K),
        "collision_rate": collision_rate(pred, K),
        "nll_kde": nll_kde(pred, gt),
    }


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap 95% CI. Returns (mean, low, high)."""
    rng = np.random.RandomState(42)
    bootstrapped = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(scores), len(scores), replace=True)
        bootstrapped.append(scores[idx].mean())

    bootstrapped = np.array(bootstrapped)
    mean = bootstrapped.mean()
    low = np.percentile(bootstrapped, (1 - ci) / 2 * 100)
    high = np.percentile(bootstrapped, (1 + ci) / 2 * 100)

    return mean, low, high
