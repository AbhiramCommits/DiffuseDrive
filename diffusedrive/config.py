"""Configuration system using dataclasses and YAML."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    source: str = "synth"  # synth | womd | av2
    num_train_scenes: int = 2000
    num_val_scenes: int = 400
    batch_size: int = 32
    num_workers: int = 4
    data_root: Optional[str] = None
    T_hist: int = 11  # frames at 10Hz = 1.0s
    T_fut: int = 80   # frames at 10Hz = 8.0s
    synthetic_seed: int = 42


@dataclass
class ModelConfig:
    model: str = "diffusion"  # diffusion | bc | mixture
    objective: str = "ddpm"  # ddpm | flow
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    polyline_dim: int = 64
    encoder_layers: int = 3
    max_agents: int = 32
    max_polylines: int = 256
    max_polyline_points: int = 30
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"


@dataclass
class TrainConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    gradient_clip_norm: float = 1.0
    log_every: int = 50
    val_every: int = 200
    checkpoint_every: int = 500
    amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 | float16
    ema_decay: float = 0.999
    seed: int = 42
    resume_from: Optional[str] = None


@dataclass
class PlannerConfig:
    planner: str = "diffusion"  # diffusion | bc | cv
    num_samples: int = 6
    sampler: str = "ddim"  # ddim | ddpm | flow
    sampler_steps: int = 20
    planning_horizon: int = 10  # steps
    replan_freq: int = 1
    ckpt_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    output_dir: str = "runs/default"


def load_config(path: str) -> ExperimentConfig:
    """Load config from YAML file and validate."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Build nested dataclass from dict
    data_cfg = DataConfig(**{k: v for k, v in raw.get("data", {}).items()
                             if k in DataConfig.__dataclass_fields__})
    model_cfg = ModelConfig(**{k: v for k, v in raw.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})
    train_cfg = TrainConfig(**{k: v for k, v in raw.get("train", {}).items()
                               if k in TrainConfig.__dataclass_fields__})
    planner_cfg = PlannerConfig(**{k: v for k, v in raw.get("planner", {}).items()
                                   if k in PlannerConfig.__dataclass_fields__})

    return ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        planner=planner_cfg,
        output_dir=raw.get("output_dir", "runs/default"),
    )


def to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Convert config to dict."""
    return asdict(cfg)


def from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    """Build config from dict."""
    return ExperimentConfig(
        data=DataConfig(**d.get("data", {})),
        model=ModelConfig(**d.get("model", {})),
        train=TrainConfig(**d.get("train", {})),
        planner=PlannerConfig(**d.get("planner", {})),
        output_dir=d.get("output_dir", "runs/default"),
    )
