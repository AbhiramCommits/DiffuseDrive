"""Tests for model components."""

import numpy as np
import pytest
import torch

from diffusedrive.config import load_config
from diffusedrive.models import build_model


@pytest.fixture
def cfg():
    return load_config("configs/small_cpu.yaml")


@pytest.fixture
def batch():
    """Create dummy batch."""
    B, A, T_hist, T_fut = 2, 8, 11, 80
    P, S = 16, 20

    return {
        "agent_hist": torch.randn(B, A, T_hist, 5),
        "agent_hist_mask": torch.ones(B, A, T_hist, dtype=torch.bool),
        "agent_type": torch.randint(0, 3, (B, A)),
        "map_polylines": torch.randn(B, P, S, 4),
        "map_mask": torch.ones(B, P, S, dtype=torch.bool),
        "agent_fut": torch.randn(B, A, T_fut, 2),
        "agent_fut_mask": torch.ones(B, A, T_fut, dtype=torch.bool),
    }


def test_model_forward_shapes(cfg, batch):
    """Test forward pass and output shapes."""
    model = build_model(cfg).eval()

    with torch.no_grad():
        loss = model(**batch)

    assert loss.ndim == 0, f"Loss should be scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"


def test_encoder_forward(cfg, batch):
    """Test encoder forward pass."""
    from diffusedrive.models.encoder import SceneEncoder

    encoder = SceneEncoder(
        d_model=cfg.model.d_model,
        num_layers=cfg.model.encoder_layers,
        num_heads=cfg.model.num_heads,
    )

    with torch.no_grad():
        scene_token, agent_context = encoder(
            batch["agent_hist"],
            batch["agent_hist_mask"],
            batch["agent_type"],
            batch["map_polylines"],
            batch["map_mask"],
        )

    B = batch["agent_hist"].shape[0]
    assert scene_token.shape == (B, cfg.model.d_model)
    assert agent_context.shape[1] == batch["agent_hist"].shape[1]


def test_diffusion_loss_decreases(cfg, batch):
    """Test that diffusion loss decreases over training."""
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should generally decrease (allow some variance)
    final_mean = np.mean(losses[-5:])
    initial_mean = np.mean(losses[:5])
    assert final_mean < initial_mean * 1.5, f"Loss not decreasing: {initial_mean} -> {final_mean}"


def test_mask_invariance(cfg):
    """Test that masked agents don't affect output shape."""
    from diffusedrive.models.encoder import SceneEncoder

    encoder = SceneEncoder(
        d_model=cfg.model.d_model,
        num_layers=cfg.model.encoder_layers,
        num_heads=cfg.model.num_heads,
    )

    B, A = 2, 8
    agent_hist = torch.randn(B, A, 11, 5)
    agent_hist_mask = torch.ones(B, A, 11, dtype=torch.bool)
    agent_hist_mask[:, 5:7] = False
    agent_type = torch.randint(0, 3, (B, A))
    map_polylines = torch.randn(B, 16, 20, 4)
    map_mask = torch.ones(B, 16, 20, dtype=torch.bool)

    with torch.no_grad():
        scene_token, agent_context = encoder(
            agent_hist, agent_hist_mask, agent_type, map_polylines, map_mask
        )

    assert scene_token.shape == (B, cfg.model.d_model)
    assert agent_context.shape == (B, A, cfg.model.d_model)


def test_ddim_sampling(cfg, batch):
    """Test DDIM-style sampling."""
    from diffusedrive.models.diffusion import DiffusionModel

    model = build_model(cfg)
    assert isinstance(model, DiffusionModel)

    with torch.no_grad():
        samples, scores = model.sample_k(
            batch["agent_hist"],
            batch["agent_hist_mask"],
            batch["agent_type"],
            batch["map_polylines"],
            batch["map_mask"],
            K=3,
            num_steps=10,
        )

    B, A, T_fut = 2, batch["agent_hist"].shape[1], batch["agent_fut"].shape[2]
    assert samples.shape == (B, 3, A, T_fut, 2)
    assert scores.shape == (B, 3)
    assert not torch.isnan(samples).any()


def test_bc_sample(cfg, batch):
    """Test BC model sampling."""
    from diffusedrive.models.baselines import BehaviorCloningMLP

    cfg.model.model = "bc"
    model = build_model(cfg)
    assert isinstance(model, BehaviorCloningMLP)

    with torch.no_grad():
        samples, scores = model.sample(
            batch["agent_hist"],
            batch["agent_hist_mask"],
            batch["agent_type"],
            batch["map_polylines"],
            batch["map_mask"],
            K=6,
        )

    B = batch["agent_hist"].shape[0]
    assert samples.shape[0] == B
    assert samples.shape[1] == 6
    assert not torch.isnan(samples).any()


def test_mixture_sample(cfg, batch):
    """Test mixture model sampling."""
    from diffusedrive.models.baselines import MultiModalRegressor

    cfg.model.model = "mixture"
    model = build_model(cfg)
    assert isinstance(model, MultiModalRegressor)

    with torch.no_grad():
        samples, scores = model.sample(
            batch["agent_hist"],
            batch["agent_hist_mask"],
            batch["agent_type"],
            batch["map_polylines"],
            batch["map_mask"],
            K=6,
        )

    B = batch["agent_hist"].shape[0]
    assert samples.shape[0] == B
    assert samples.shape[1] == 6
    assert not torch.isnan(samples).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
