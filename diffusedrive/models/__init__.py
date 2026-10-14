"""Model building and selection."""

import torch.nn as nn
from .encoder import SceneEncoder
from .diffusion import DiffusionModel, DenoiserTransformer
from .baselines import BehaviorCloningMLP, MultiModalRegressor


def build_model(cfg) -> nn.Module:
    """Build model based on config."""
    
    # Build encoder
    encoder = SceneEncoder(
        d_agent=5,
        d_polyline=4,
        d_model=cfg.model.d_model,
        d_polyline_enc=cfg.model.polyline_dim,
        num_layers=cfg.model.encoder_layers,
        num_heads=cfg.model.num_heads,
        max_agents=cfg.model.max_agents,
        max_polylines=cfg.model.max_polylines,
    )
    
    if cfg.model.model == "diffusion":
        # Build denoiser
        denoiser = DenoiserTransformer(
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            ff_dim=cfg.model.ff_dim,
            dropout=cfg.model.dropout,
            max_agents=cfg.model.max_agents,
            T_fut=cfg.data.T_fut,
            objective=cfg.model.objective,
        )
        
        model = DiffusionModel(
            encoder=encoder,
            denoiser=denoiser,
            timesteps=cfg.model.diffusion_steps,
            objective=cfg.model.objective,
        )
    
    elif cfg.model.model == "bc":
        model = BehaviorCloningMLP(
            encoder=encoder,
            d_model=cfg.model.d_model,
            T_fut=cfg.data.T_fut,
            max_agents=cfg.model.max_agents,
        )
    
    elif cfg.model.model == "mixture":
        model = MultiModalRegressor(
            encoder=encoder,
            d_model=cfg.model.d_model,
            T_fut=cfg.data.T_fut,
            max_agents=cfg.model.max_agents,
            num_modes=6,
        )
    
    else:
        raise ValueError(f"Unknown model: {cfg.model.model}")
    
    return model


__all__ = [
    "SceneEncoder",
    "DiffusionModel",
    "DenoiserTransformer",
    "BehaviorCloningMLP",
    "MultiModalRegressor",
    "build_model",
]
