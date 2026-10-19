"""Python fallback closed-loop simulator."""

from typing import Tuple

import numpy as np


class BicycleModel:
    """Simple bicycle kinematic model."""

    def __init__(self, length: float = 0.5, dt: float = 0.1):
        self.length = length
        self.dt = dt

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Step bicycle model. state: [x, y, theta, v]. action: [accel, steer]."""
        x, y, theta, v = state
        accel, steer = action

        # Update velocity
        v_new = np.clip(v + accel * self.dt, -5, 20)

        # Bicycle kinematic
        dx = v_new * np.cos(theta) * self.dt
        dy = v_new * np.sin(theta) * self.dt
        dtheta = (v_new / self.length) * np.tan(steer) * self.dt
        theta_new = theta + dtheta

        return np.array([x + dx, y + dy, theta_new, v_new])


def check_collision(ego_pos: np.ndarray, other_pos: np.ndarray, radius: float = 0.5) -> bool:
    """Circle-based collision check."""
    return np.linalg.norm(ego_pos - other_pos) < 2 * radius


def check_offroad(pos: np.ndarray, map_bounds: Tuple[float, float, float, float]) -> bool:
    """Check if position is off map bounds."""
    x_min, x_max, y_min, y_max = map_bounds
    return pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max
