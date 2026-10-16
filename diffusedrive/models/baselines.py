"""Baseline forecasters: BC and mixture models."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SceneEncoder
from .layers import MLP


class BehaviorCloningMLP(nn.Module):
    """Simple deterministic forecaster via behavior cloning."""

    def __init__(
        self,
        encoder: SceneEncoder,
        d_model: int = 128,
        T_fut: int = 80,
        max_agents: int = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.T_fut = T_fut
        self.max_agents = max_agents

        # Per-agent decoder: context -> T_fut * 2
        self.decoder = MLP(d_model, d_model * 2, T_fut * 2)

    def forward(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        agent_fut: torch.Tensor,
        agent_fut_mask: torch.Tensor,
    ) -> torch.Tensor:
        """L2 regression loss."""
        B, A, T_fut, _ = agent_fut.shape

        _, agent_context = self.encoder(
            agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
        )

        # Decode trajectories
        pred = self.decoder(agent_context)  # [B, A, T_fut*2]
        pred = pred.reshape(B, A, T_fut, 2)

        # MSE loss
        loss = F.mse_loss(pred, agent_fut, reduction='none')
        loss = loss * agent_fut_mask.unsqueeze(-1)

        return loss.mean()

    def sample(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        K: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample K trajectories (deterministic, all same)."""
        B, A, _, _ = agent_hist.shape

        with torch.no_grad():
            _, agent_context = self.encoder(
                agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
            )

            pred = self.decoder(agent_context)
            pred = pred.reshape(B, A, self.T_fut, 2)

        # Repeat K times (deterministic)
        samples = pred.unsqueeze(1).expand(B, K, A, self.T_fut, 2)
        scores = torch.zeros(B, K, device=samples.device)

        return samples, scores


class MultiModalRegressor(nn.Module):
    """Multi-head mixture model for multimodal forecasting."""

    def __init__(
        self,
        encoder: SceneEncoder,
        d_model: int = 128,
        T_fut: int = 80,
        max_agents: int = 32,
        num_modes: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.T_fut = T_fut
        self.max_agents = max_agents
        self.num_modes = num_modes

        # K trajectory heads
        self.trajectory_heads = nn.ModuleList([
            MLP(d_model, d_model * 2, T_fut * 2)
            for _ in range(num_modes)
        ])

        # Mode logits
        self.mode_logits = MLP(d_model, d_model, num_modes)

    def forward(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        agent_fut: torch.Tensor,
        agent_fut_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Winner-takes-all loss."""
        B, A, T_fut, _ = agent_fut.shape

        _, agent_context = self.encoder(
            agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
        )

        # Get mode logits
        logits = self.mode_logits(agent_context)  # [B, A, num_modes]
        mode_probs = F.softmax(logits, dim=-1)

        # Get K trajectory predictions
        traj_preds = []
        for head in self.trajectory_heads:
            pred = head(agent_context)  # [B, A, T_fut*2]
            pred = pred.reshape(B, A, T_fut, 2)
            traj_preds.append(pred)

        traj_preds = torch.stack(traj_preds, dim=1)  # [B, num_modes, A, T_fut, 2]

        # Compute loss for each mode
        gt = agent_fut.unsqueeze(1)  # [B, 1, A, T_fut, 2]
        loss_per_mode = F.mse_loss(traj_preds, gt, reduction='none')  # [B, num_modes, A, T, 2]

        # Mask
        loss_per_mode = loss_per_mode * agent_fut_mask.unsqueeze(1).unsqueeze(-1)
        loss_per_mode = loss_per_mode.mean(dim=(2, 3, 4))  # [B, num_modes]

        # Winner-takes-all: select best mode for each batch
        best_mode = loss_per_mode.argmin(dim=1)  # [B]
        best_loss = loss_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1)

        return best_loss.mean()

    def sample(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        K: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample K trajectories."""
        B, A, _, _ = agent_hist.shape

        with torch.no_grad():
            _, agent_context = self.encoder(
                agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
            )

            # Get mode logits
            logits = self.mode_logits(agent_context)  # [B, A, num_modes]

            # Get K predictions
            traj_preds = []
            for head in self.trajectory_heads[:K]:
                pred = head(agent_context)
                pred = pred.reshape(B, A, self.T_fut, 2)
                traj_preds.append(pred)

            samples = torch.stack(traj_preds, dim=1)  # [B, K, A, T_fut, 2]

            # Scores: mode logits softmax (as confidence)
            scores = F.softmax(logits, dim=-1).max(dim=-1)[0]  # [B, A]
            scores = scores.mean(dim=1, keepdim=True).expand(B, K)  # [B, K]

        return samples, scores
