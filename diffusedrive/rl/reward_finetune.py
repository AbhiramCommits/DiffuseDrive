"""Reward-guided fine-tuning (DDPO-style)."""

import json
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from diffusedrive.config import load_config
from diffusedrive.data import build_dataset, collate_batch
from diffusedrive.models import build_model
from diffusedrive.plan.cost import total_cost
from diffusedrive.train.loop import save_checkpoint


def finetune(cfg_path: str, ckpt_path: str, output_dir: str, iterations: int = 100):
    """Fine-tune with reward guidance."""

    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and checkpoint
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Create reference model (frozen)
    ref_model = build_model(cfg).to(device)
    ref_model.load_state_dict(ckpt["model_state"])
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = Adam(model.parameters(), lr=1e-4)
    dataset = build_dataset(cfg.data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_batch,
        shuffle=True,
    )

    metrics_file = output_path / "rl_metrics.jsonl"

    model_keys = {
        "agent_hist", "agent_hist_mask", "agent_fut", "agent_fut_mask",
        "agent_type", "map_polylines", "map_mask"
    }

    for iteration in tqdm(range(iterations)):
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_batch = {k: v for k, v in batch.items() if k in model_keys}

            # Sample trajectories
            with torch.no_grad():
                ref_samples, _ = ref_model.sample_k(
                    model_batch["agent_hist"],
                    model_batch["agent_hist_mask"],
                    model_batch["agent_type"],
                    model_batch["map_polylines"],
                    model_batch["map_mask"],
                    K=4,
                )

            samples, _ = model.sample_k(
                model_batch["agent_hist"],
                model_batch["agent_hist_mask"],
                model_batch["agent_type"],
                model_batch["map_polylines"],
                model_batch["map_mask"],
                K=4,
            )

            # Compute rewards (negative cost)
            gt_fut = model_batch["agent_fut"].cpu().numpy()
            samples = samples.cpu().detach().numpy()

            rewards = []
            for b in range(samples.shape[0]):
                ego_traj = samples[b, 0, 0, :, :]  # First sample, ego agent
                other_trajs = gt_fut[b, 1:, :]
                goal = gt_fut[b, 0, -1]

                cost = total_cost(ego_traj, other_trajs.reshape(-1, gt_fut.shape[2], 2), goal, (-100, 100, -100, 100))
                rewards.append(-cost)  # Negative cost = reward

            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

            # Policy gradient: maximize expected reward (dummy loss for now)
            loss = -(rewards.mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                metrics = {"iteration": iteration, "mean_reward": rewards.mean().item()}
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

    # Save checkpoint
    save_checkpoint(model, optimizer, None, None, None, 0, 0, 0.0, output_path / "ckpt_best.pt")
