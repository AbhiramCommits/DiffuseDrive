"""Planning cost functions."""

import numpy as np


def collision_cost(ego_traj: np.ndarray, other_trajs: np.ndarray, radius: float = 0.5) -> float:
    """Collision cost: 1 if collision, 0 else. ego_traj: [T, 2], other_trajs: [A-1, T, 2]"""
    for t in range(ego_traj.shape[0]):
        for a in range(other_trajs.shape[0]):
            dist = np.linalg.norm(ego_traj[t] - other_trajs[a, t])
            if dist < 2 * radius:
                return 1.0
    return 0.0


def progress_cost(ego_traj: np.ndarray, goal: np.ndarray) -> float:
    """Negative progress: -distance to goal. ego_traj: [T, 2]"""
    final_dist = np.linalg.norm(ego_traj[-1] - goal)
    return final_dist


def comfort_cost(ego_traj: np.ndarray) -> float:
    """Jerk (3rd derivative) penalty. ego_traj: [T, 2]"""
    if len(ego_traj) < 4:
        return 0.0

    vel = np.diff(ego_traj, axis=0)
    accel = np.diff(vel, axis=0)
    jerk = np.diff(accel, axis=0)

    return np.linalg.norm(jerk) / max(1.0, len(jerk))


def offroad_cost(ego_traj: np.ndarray, map_bounds: tuple) -> float:
    """Offroad penalty: 1 if any offroad, 0 else."""
    x_min, x_max, y_min, y_max = map_bounds

    for pos in ego_traj:
        if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
            return 1.0

    return 0.0


def total_cost(
    ego_traj: np.ndarray,
    other_trajs: np.ndarray,
    goal: np.ndarray,
    map_bounds: tuple,
    weights: dict = None,
) -> float:
    """Weighted sum of costs."""
    if weights is None:
        weights = {
            "collision": 10.0,
            "progress": 1.0,
            "comfort": 0.1,
            "offroad": 5.0,
        }

    cost = (
        weights["collision"] * collision_cost(ego_traj, other_trajs) +
        weights["progress"] * progress_cost(ego_traj, goal) +
        weights["comfort"] * comfort_cost(ego_traj) +
        weights["offroad"] * offroad_cost(ego_traj, map_bounds)
    )

    return cost
