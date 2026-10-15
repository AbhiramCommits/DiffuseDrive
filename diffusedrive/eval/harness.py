"""Evaluation harness."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from diffusedrive.config import load_config
from diffusedrive.data import build_dataset, collate_batch
from diffusedrive.models import build_model
from .metrics import compute_metrics, bootstrap_ci


def evaluate(
    ckpt_path: str,
    cfg_path: str,
    K: int = 6,
    num_steps: int = 50,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Evaluate model on val set. Return metrics dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load config and model
    cfg = load_config(cfg_path)
    model = build_model(cfg).to(device).eval()

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # Load val dataset
    val_dataset = build_dataset(cfg.data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=collate_batch,
        shuffle=False,
    )

    all_metrics = {k: [] for k in ["minADE_k", "minFDE_k", "miss_rate", "collision_rate"]}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            model_keys = {
                "agent_hist", "agent_hist_mask", "agent_fut", "agent_fut_mask",
                "agent_type", "map_polylines", "map_mask"
            }
            model_batch = {k: v for k, v in batch.items() if k in model_keys}

            # Sample K futures
            if hasattr(model, "sample_k"):
                samples, _ = model.sample_k(
                    model_batch["agent_hist"],
                    model_batch["agent_hist_mask"],
                    model_batch["agent_type"],
                    model_batch["map_polylines"],
                    model_batch["map_mask"],
                    K=K,
                    num_steps=num_steps,
                )
            else:
                samples, _ = model.sample(
                    model_batch["agent_hist"],
                    model_batch["agent_hist_mask"],
                    model_batch["agent_type"],
                    model_batch["map_polylines"],
                    model_batch["map_mask"],
                    K=K,
                )

            # Compute metrics per batch
            samples = samples.cpu().numpy()
            gt = model_batch["agent_fut"].cpu().numpy()

            for b in range(samples.shape[0]):
                metrics = compute_metrics(samples[b], gt[b], K)
                for k in all_metrics:
                    if k in metrics:
                        all_metrics[k].append(metrics[k])

    # Aggregate with bootstrap CI
    result = {}
    for k, v in all_metrics.items():
        if v:
            v = np.array(v)
            mean, low, high = bootstrap_ci(v)
            result[k] = {"mean": float(mean), "ci_low": float(low), "ci_high": float(high)}

    return result
