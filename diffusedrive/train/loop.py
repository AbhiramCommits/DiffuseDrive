"""Training loop with AMP, checkpointing, resume."""

import json
import math
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def restore(self):
        """Restore EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.decay = state["decay"]


def create_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Create cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[ExponentialMovingAverage],
    cfg,
    device: torch.device,
    epoch: int,
    global_step: int,
    metrics_file: Path,
) -> Tuple[float, int]:
    """Train one epoch, return avg loss and steps."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Filter batch to model inputs only
        model_keys = {
            "agent_hist", "agent_hist_mask", "agent_fut", "agent_fut_mask",
            "agent_type", "map_polylines", "map_mask"
        }
        model_batch = {k: v for k, v in batch.items() if k in model_keys}

        optimizer.zero_grad()

        # Forward with AMP
        if cfg.train.amp:
            with torch.cuda.amp.autocast(dtype=torch.float16 if cfg.train.amp_dtype == "float16" else torch.bfloat16):
                loss = model(**model_batch)
        else:
            loss = model(**model_batch)

        # Backward
        if cfg.train.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Gradient clipping
        if cfg.train.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)

        # Optimizer step
        if cfg.train.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        # Log metrics
        if global_step % cfg.train.log_every == 0:
            metrics = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    cfg,
    device: torch.device,
) -> float:
    """Validate, return avg loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    model_keys = {
        "agent_hist", "agent_hist_mask", "agent_fut", "agent_fut_mask",
        "agent_type", "map_polylines", "map_mask"
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_batch = {k: v for k, v in batch.items() if k in model_keys}
            loss = model(**model_batch)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[ExponentialMovingAverage],
    epoch: int,
    step: int,
    val_loss: float,
    path: Path,
):
    """Save checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "val_loss": val_loss,
    }

    if ema is not None:
        checkpoint["ema_state"] = ema.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[ExponentialMovingAverage],
    path: Path,
):
    """Load checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    if scaler is not None and checkpoint["scaler_state"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    if ema is not None and "ema_state" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state"])

    logger.info(f"Loaded checkpoint from {path}")
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0), checkpoint.get("val_loss", float("inf"))
