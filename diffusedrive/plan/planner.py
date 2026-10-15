"""Sampling-based and baseline planners."""

import numpy as np
import torch
from typing import Tuple
from .cost import total_cost


class SamplingPlanner:
    """Sample K futures, score, pick best."""

    def __init__(self, model, planning_horizon: int = 10, replan_freq: int = 1):
        self.model = model
        self.planning_horizon = planning_horizon
        self.replan_freq = replan_freq
        self.device = next(model.parameters()).device

    def plan(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        ego_idx: int = 0,
        K: int = 6,
        goal: np.ndarray = None,
    ) -> np.ndarray:
        """Plan ego trajectory. Return [planning_horizon, 2]"""

        with torch.no_grad():
            samples, _ = self.model.sample_k(
                agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask,
                K=K, num_steps=10,
            )

        # samples: [B, K, A, T_fut, 2]
        samples = samples[0].cpu().numpy()  # [K, A, T_fut, 2]

        # Score each future for ego agent
        scores = []
        ego_trajs = samples[:, ego_idx, :self.planning_horizon]

        for k in range(K):
            other_trajs = np.concatenate([
                samples[k, :ego_idx, :self.planning_horizon],
                samples[k, ego_idx+1:, :self.planning_horizon],
            ], axis=0)

            if goal is None:
                goal = ego_trajs[k, -1]  # Final position as goal

            cost = total_cost(ego_trajs[k], other_trajs, goal, (-100, 100, -100, 100))
            scores.append(cost)

        # Pick lowest cost
        best_k = np.argmin(scores)
        return ego_trajs[best_k]


class BCPlanner:
    """Execute BC model output (deterministic)."""

    def __init__(self, model, planning_horizon: int = 10):
        self.model = model
        self.planning_horizon = planning_horizon
        self.device = next(model.parameters()).device

    def plan(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        ego_idx: int = 0,
    ) -> np.ndarray:
        """Return [planning_horizon, 2]"""

        with torch.no_grad():
            samples, _ = self.model.sample(
                agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask, K=1
            )

        samples = samples[0, 0, ego_idx, :self.planning_horizon].cpu().numpy()
        return samples


class ConstantVelocityPlanner:
    """Naive constant velocity baseline."""

    def __init__(self, planning_horizon: int = 10):
        self.planning_horizon = planning_horizon

    def plan(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        ego_idx: int = 0,
    ) -> np.ndarray:
        """Extrapolate with constant velocity."""

        agent_hist = agent_hist.cpu().numpy()
        ego_hist = agent_hist[0, ego_idx]  # [T_hist, 5]

        # Last two positions to estimate velocity
        pos_prev = ego_hist[-2, :2]
        pos_curr = ego_hist[-1, :2]
        vel = (pos_curr - pos_prev) / 0.1

        # Extrapolate
        traj = []
        pos = pos_curr.copy()
        for _ in range(self.planning_horizon):
            pos = pos + vel * 0.1
            traj.append(pos.copy())

        return np.array(traj)
