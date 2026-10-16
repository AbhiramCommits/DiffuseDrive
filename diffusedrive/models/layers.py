"""Basic layers: MLP, normalization, embeddings, attention, FiLM."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple feedforward MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / norm) * self.weight


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embeddings (for time steps or positions)."""
    assert dim % 2 == 0
    half_dim = dim // 2

    # x: [...] -> [..., dim]
    x_flat = x.flatten()
    freqs = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=x.device) *
        (-math.log(10000) / (half_dim - 1))
    )

    emb = x_flat.unsqueeze(-1) * freqs.unsqueeze(0)  # [N, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [N, dim]

    # Reshape back
    shape = list(x.shape) + [dim]
    return emb.reshape(shape)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional key padding mask."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, d_model]
            key: [batch, seq_k, d_model]
            value: [batch, seq_v, d_model]
            key_padding_mask: [batch, seq_k], True for positions to mask
        Returns:
            output: [batch, seq_q, d_model]
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # Linear projections
        Q = self.W_q(query)  # [batch, seq_q, d_model]
        K = self.W_k(key)    # [batch, seq_k, d_model]
        V = self.W_v(value)  # [batch, seq_v, d_model]

        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_k] -> [batch, 1, 1, seq_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_q, d_k]
        output = output.transpose(1, 2).reshape(batch_size, seq_q, self.d_model)
        output = self.W_o(output)

        return output


class FiLMBlock(nn.Module):
    """FiLM (Feature-wise Linear Modulation) conditioning block."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.gamma_mlp = nn.Linear(d_cond, d_model)
        self.beta_mlp = nn.Linear(d_cond, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
            cond: [batch, d_cond]
        Returns:
            modulated: [batch, seq, d_model]
        """
        gamma = self.gamma_mlp(cond)  # [batch, d_model]
        beta = self.beta_mlp(cond)    # [batch, d_model]

        # Broadcast to sequence dimension
        gamma = gamma.unsqueeze(1)  # [batch, 1, d_model]
        beta = beta.unsqueeze(1)    # [batch, 1, d_model]

        return x * (1 + gamma) + beta
