"""Training CLI entrypoint."""

import argparse
import json
import logging
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from diffusedrive.config import load_config
from diffusedrive.data import build_dataset, collate_batch
from diffusedrive.models import build_model
from diffusedrive.utils.seed import set_seed
from .loop import (
    train_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    ExponentialMovingAverage,
    create_cosine_schedule_with_warmup,
)
from .ddp import init_distributed, is_main_process, get_rank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--model", default="diffusion", help="Model: diffusion, bc, mixture")
    parser.add_argument("--out", default="runs/default", help="Output directory")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup distributed
    rank, world_size = init_distributed()

    # Load config
    cfg = load_config(args.config)
    cfg.model.model = args.model
    cfg.output_dir = args.out
    cfg.train.resume_from = args.resume_from

    # Create output dir (rank 0 only)
    out_path = Path(cfg.output_dir)
    if is_main_process():
        out_path.mkdir(parents=True, exist_ok=True)

        # Save config
        import yaml
        with open(out_path / "config.yaml", "w") as f:
            yaml.dump({
                "data": cfg.data.__dict__,
                "model": cfg.model.__dict__,
                "train": cfg.train.__dict__,
            }, f)

    # Set seed
    set_seed(cfg.train.seed)

    # Build dataset
    logger.info(f"Building dataset ({cfg.data.source})")
    train_dataset = build_dataset(cfg.data)

    # Build validation dataset
    val_dataset = build_dataset(cfg.data)

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    # Build model
    logger.info(f"Building {cfg.model.model} model")
    model = build_model(cfg)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    num_training_steps = len(train_loader) * cfg.train.num_epochs
    scheduler = create_cosine_schedule_with_warmup(
        optimizer,
        cfg.train.warmup_steps,
        num_training_steps,
    )

    # AMP
    scaler = None
    if cfg.train.amp:
        scaler = torch.cuda.amp.GradScaler()

    # EMA
    ema = ExponentialMovingAverage(model, cfg.train.ema_decay) if cfg.train.ema_decay < 1.0 else None

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if cfg.train.resume_from:
        logger.info(f"Resuming from {cfg.train.resume_from}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, scaler, ema, Path(cfg.train.resume_from)
        )

    # Training loop
    logger.info(f"Starting training for {cfg.train.num_epochs} epochs")

    metrics_file = out_path / "metrics.jsonl"
    if is_main_process():
        metrics_file.touch()

    for epoch in range(start_epoch, cfg.train.num_epochs):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema, cfg, device,
            epoch, global_step, metrics_file,
        )

        logger.info(f"Epoch {epoch + 1} train loss: {train_loss:.4f}")

        # Validate
        if is_main_process() and (epoch + 1) % (cfg.train.val_every // len(train_loader) + 1) == 0:
            val_loss = validate(model, val_loader, cfg, device)
            logger.info(f"Epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, ema,
                    epoch, global_step, val_loss,
                    out_path / "ckpt_best.pt",
                )

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, ema,
                    epoch, global_step, val_loss,
                    out_path / f"ckpt_epoch_{epoch + 1:03d}.pt",
                )

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
