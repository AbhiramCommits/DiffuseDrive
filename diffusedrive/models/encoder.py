"""Scene encoder: polyline encoding and agent-map interaction."""

from typing import Tuple

import torch
import torch.nn as nn

from .layers import MLP, MultiHeadAttention, RMSNorm


class PolylineEncoder(nn.Module):
    """PointNet-style encoding of map polylines."""

    def __init__(self, d_polyline: int = 4, d_out: int = 64):
        super().__init__()
        self.mlp1 = MLP(d_polyline, 64, d_out)
        self.mlp2 = MLP(d_out, 128, d_out)

    def forward(self, polylines: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            polylines: [batch, num_polylines, max_points, 4]
            mask: [batch, num_polylines, max_points], True for valid
        Returns:
            encoded: [batch, num_polylines, d_out]
        """
        B, P, S, _ = polylines.shape

        # MLP over features
        x = self.mlp1(polylines)  # [B, P, S, d_out]
        x = self.mlp2(x)          # [B, P, S, d_out]

        # Max-pool over points for each polyline
        x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        x = torch.amax(x, dim=2)  # [B, P, d_out]

        # Clamp to avoid inf
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x


class AgentHistoryEncoder(nn.Module):
    """Encode agent histories using 1D convolution."""

    def __init__(self, d_hist: int = 5, d_out: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(d_hist, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, d_out, kernel_size=3, padding=1)

    def forward(self, hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hist: [batch, num_agents, T_hist, 5]
            mask: [batch, num_agents, T_hist], True for valid
        Returns:
            encoded: [batch, num_agents, d_out]
        """
        B, A, T, _ = hist.shape

        # Reshape for conv1d: [B*A, 5, T]
        x = hist.reshape(B * A, -1, T)  # [B*A, 5, T]
        x = torch.relu(self.conv(x))
        x = torch.relu(self.conv2(x))  # [B*A, d_out, T]

        # Global max pool
        x = x.max(dim=2)[0]  # [B*A, d_out]
        x = x.reshape(B, A, -1)  # [B, A, d_out]

        return x


class SceneEncoder(nn.Module):
    """Encode scene: agents + map with interaction layers."""

    def __init__(
        self,
        d_agent: int = 5,
        d_polyline: int = 4,
        d_model: int = 128,
        d_polyline_enc: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        max_agents: int = 32,
        max_polylines: int = 256,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Agent history encoding
        self.agent_hist_encoder = AgentHistoryEncoder(d_agent, d_model)

        # Agent type embedding
        self.agent_type_embed = nn.Embedding(3, d_model)

        # Polyline encoding
        self.polyline_encoder = PolylineEncoder(d_polyline, d_polyline_enc)
        self.polyline_proj = nn.Linear(d_polyline_enc, d_model)

        # Interaction layers: alternating agent-agent and agent-map attention
        self.agent_self_attn = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.agent_map_cross_attn = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.agent_norm1 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.agent_norm2 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.agent_ff = nn.ModuleList([
            MLP(d_model, d_model * 4, d_model)
            for _ in range(num_layers)
        ])
        self.agent_norm3 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])

    def forward(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_hist: [batch, max_agents, T_hist, 5]
            agent_hist_mask: [batch, max_agents, T_hist], True for valid
            agent_type: [batch, max_agents]
            map_polylines: [batch, max_polylines, max_points, 4]
            map_mask: [batch, max_polylines, max_points], True for valid
        Returns:
            scene_tokens: [batch, d_model]
            agent_context: [batch, max_agents, d_model]
        """
        B, A, _, _ = agent_hist.shape

        # Encode agent histories
        agent_emb = self.agent_hist_encoder(agent_hist, agent_hist_mask)  # [B, A, d_model]

        # Add agent type embeddings
        type_emb = self.agent_type_embed(agent_type)  # [B, A, d_model]
        agent_emb = agent_emb + type_emb

        # Encode polylines
        map_emb = self.polyline_encoder(map_polylines, map_mask)  # [B, P, d_polyline_enc]
        map_emb = self.polyline_proj(map_emb)  # [B, P, d_model]

        # Create agent mask: [B, A], False for padded agents
        agent_mask = agent_hist_mask.any(dim=2)  # [B, A]
        agent_key_mask = ~agent_mask  # True = mask

        # Create polyline mask: [B, P], True for valid polylines
        poly_mask = map_mask.any(dim=2)  # [B, P]
        poly_key_mask = ~poly_mask  # True = mask

        # Interaction layers
        for layer in range(self.num_layers):
            # Agent self-attention
            agent_attn = self.agent_self_attn[layer](
                agent_emb, agent_emb, agent_emb,
                key_padding_mask=agent_key_mask
            )
            agent_emb = self.agent_norm1[layer](agent_emb + agent_attn)

            # Agent-map cross-attention
            map_attn = self.agent_map_cross_attn[layer](
                agent_emb, map_emb, map_emb,
                key_padding_mask=poly_key_mask
            )
            agent_emb = self.agent_norm2[layer](agent_emb + map_attn)

            # FFN
            ff_out = self.agent_ff[layer](agent_emb)
            agent_emb = self.agent_norm3[layer](agent_emb + ff_out)

        # Scene token: mean over agents (excluding padded)
        scene_token = (agent_emb * agent_mask.unsqueeze(-1)).sum(dim=1) / agent_mask.sum(dim=1, keepdim=True).clamp(min=1)

        return scene_token, agent_emb
