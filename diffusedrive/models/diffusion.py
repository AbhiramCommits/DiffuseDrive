"""Diffusion model: schedule, denoising, sampling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .layers import sinusoidal_embedding, RMSNorm, MultiHeadAttention, MLP, FiLMBlock


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for beta values."""
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple) -> torch.Tensor:
    """Extract values from array at timesteps and broadcast."""
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape)


class DiffusionSchedule:
    """Diffusion schedule (alpha, beta, etc.)."""

    def __init__(self, timesteps: int = 1000):
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # Posterior variance
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise."""
        shape = x_0.shape
        sqrt_alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, shape)
        sqrt_one_minus_alpha = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, shape)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


class DenoiserTransformer(nn.Module):
    """Transformer denoiser conditioned on scene and timestep."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_agents: int = 32,
        T_fut: int = 80,
        objective: str = "epsilon",
    ):
        super().__init__()

        self.d_model = d_model
        self.max_agents = max_agents
        self.T_fut = T_fut
        self.objective = objective

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Trajectory input embedding: [A, T_fut, 2] -> token sequence
        self.traj_embed = nn.Linear(2, d_model)

        # Transformer layers with FiLM conditioning
        self.transformer_layers = nn.ModuleList()
        self.film_blocks = nn.ModuleList()

        for _ in range(num_layers):
            self.transformer_layers.append(nn.ModuleDict({
                'attn': MultiHeadAttention(d_model, num_heads, dropout),
                'norm1': RMSNorm(d_model),
                'ff': MLP(d_model, ff_dim, d_model, dropout),
                'norm2': RMSNorm(d_model),
            }))
            self.film_blocks.append(FiLMBlock(d_model, d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        scene_cond: torch.Tensor,
        agent_context: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [batch, max_agents, T_fut, 2] noisy trajectory
            t: [batch] timesteps
            scene_cond: [batch, d_model] scene token
            agent_context: [batch, max_agents, d_model] per-agent context
            agent_mask: [batch, max_agents] True for valid agents
        Returns:
            pred: [batch, max_agents, T_fut, 2] predicted noise or velocity
        """
        B, A, T, _ = x_t.shape

        # Embed timestep
        t_emb = sinusoidal_embedding(t.float(), self.d_model)  # [B, d_model]
        t_emb = self.time_embed(t_emb)  # [B, d_model]

        # Combine scene and time conditioning
        cond = scene_cond + t_emb  # [B, d_model]

        # Embed trajectory: [B, A, T, 2] -> [B, A, T, d_model]
        x_emb = self.traj_embed(x_t)  # [B, A, T, d_model]

        # Add agent context to each timestep
        x_emb = x_emb + agent_context.unsqueeze(2)  # [B, A, T, d_model]

        # Flatten to sequence: [B, A*T, d_model]
        x_flat = x_emb.reshape(B, A * T, self.d_model)

        # Create agent mask for attention (mask padded agents across all timesteps)
        if agent_mask is not None:
            attn_mask = agent_mask.unsqueeze(2).expand(B, A, T).reshape(B, A * T)
            attn_mask = ~attn_mask  # True = mask (inverted for attention)
        else:
            attn_mask = None

        # Transformer with FiLM conditioning
        for layer, film in zip(self.transformer_layers, self.film_blocks):
            # Self-attention
            attn_out = layer['attn'](x_flat, x_flat, x_flat, key_padding_mask=attn_mask)
            x_flat = layer['norm1'](x_flat + attn_out)

            # FiLM modulation
            x_flat = film(x_flat, cond)

            # FFN
            ff_out = layer['ff'](x_flat)
            x_flat = layer['norm2'](x_flat + ff_out)

        # Project to 2D
        pred = self.out_proj(x_flat)  # [B, A*T, 2]
        pred = pred.reshape(B, A, T, 2)

        return pred


class DiffusionModel(nn.Module):
    """Full diffusion model."""

    def __init__(
        self,
        encoder: nn.Module,
        denoiser: DenoiserTransformer,
        timesteps: int = 1000,
        objective: str = "epsilon",
    ):
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.schedule = DiffusionSchedule(timesteps)
        self.objective = objective

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
        """DDPM loss."""
        B, A, T_fut, _ = agent_fut.shape
        device = agent_fut.device

        # Encode scene
        scene_cond, agent_context = self.encoder(
            agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
        )

        # Sample random timesteps
        t = torch.randint(0, len(self.schedule.betas), (B,), device=device)

        # Sample random noise
        noise = torch.randn_like(agent_fut)

        # Forward diffusion
        x_t = self.schedule.q_sample(agent_fut, t, noise)

        # Predict noise
        agent_mask = agent_hist_mask.any(dim=2)  # [B, A]
        pred = self.denoiser(x_t, t, scene_cond, agent_context, agent_mask)

        # MSE loss
        loss = F.mse_loss(pred, noise, reduction='none')

        # Mask padded agents
        loss = loss * agent_fut_mask.unsqueeze(-1)

        return loss.mean()

    def sample(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        num_steps: int = 1000,
    ) -> torch.Tensor:
        """DDPM sampling."""
        B, A, _, _ = agent_hist.shape
        device = agent_hist.device

        # Encode scene
        scene_cond, agent_context = self.encoder(
            agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
        )

        # Start from noise
        x_t = torch.randn(B, A, self.denoiser.T_fut, 2, device=device)
        agent_mask = agent_hist_mask.any(dim=2)

        # Reverse diffusion
        step_size = len(self.schedule.betas) // num_steps
        for i in range(len(self.schedule.betas) - 1, 0, -step_size):
            t = torch.full((B,), i, dtype=torch.long, device=device)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.denoiser(x_t, t, scene_cond, agent_context, agent_mask)

            # DDPM step
            alpha = self.schedule.alphas[i]
            alpha_prev = self.schedule.alphas[i - step_size]

            x_0_pred = (x_t - torch.sqrt(1 - self.schedule.alphas_cumprod[i]) * noise_pred) / torch.sqrt(self.schedule.alphas_cumprod[i])
            x_0_pred = torch.clamp(x_0_pred, -20, 20)

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = (
                    torch.sqrt(alpha_prev) * x_0_pred +
                    torch.sqrt(1 - alpha_prev) * noise
                )
            else:
                x_t = x_0_pred

        return x_t

    def sample_k(
        self,
        agent_hist: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        agent_type: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        K: int = 6,
        num_steps: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample K diverse futures."""
        samples = []

        for _ in range(K):
            sample = self.sample(
                agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask, num_steps
            )
            samples.append(sample)

        samples = torch.stack(samples, dim=1)  # [B, K, A, T_fut, 2]

        # Score placeholder (uniform for now)
        scores = torch.zeros(samples.shape[0], K, device=samples.device)

        return samples, scores
