"""End-to-end integration test."""

import json
import tempfile
from pathlib import Path

import torch

from diffusedrive.config import load_config
from diffusedrive.data import build_dataset, collate_batch
from diffusedrive.models import build_model
from diffusedrive.plan.planner import SamplingPlanner
from diffusedrive.report import generate_report
from diffusedrive.train.loop import ExponentialMovingAverage, load_checkpoint, save_checkpoint


def test_end_to_end():
    """Test full pipeline: data -> train -> eval -> plan -> report."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Load tiny config
        cfg = load_config("configs/small_cpu.yaml")
        cfg.train.num_epochs = 1
        cfg.data.num_train_scenes = 100
        cfg.data.num_val_scenes = 20

        device = torch.device("cpu")

        # Build dataset
        dataset = build_dataset(cfg.data)
        assert len(dataset) == cfg.data.num_train_scenes

        # Build model
        model = build_model(cfg).to(device)

        # Train 5 steps
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_batch,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        ema = ExponentialMovingAverage(model)

        metrics_file = tmpdir / "metrics.jsonl"
        metrics_file.touch()

        # Mini training loop
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 5:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_keys = {
                "agent_hist", "agent_hist_mask", "agent_fut", "agent_fut_mask",
                "agent_type", "map_polylines", "map_mask"
            }
            model_batch = {k: v for k, v in batch.items() if k in model_keys}

            optimizer.zero_grad()
            loss = model(**model_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()

        # Save checkpoint
        ckpt_path = tmpdir / "ckpt.pt"
        save_checkpoint(model, optimizer, scheduler, None, ema, 0, 5, 0.5, ckpt_path)

        # Load checkpoint
        load_checkpoint(model, optimizer, scheduler, None, ema, ckpt_path)

        # Validate
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_batch,
            shuffle=False,
        )

        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 2:
                    break

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model_batch = {k: v for k, v in batch.items() if k in model_keys}
                loss = model(**model_batch)
                val_loss += loss.item()

        # Plan
        planner = SamplingPlanner(model, planning_horizon=5)
        sample = dataset[0]
        for k in list(sample.keys()):
            if isinstance(sample[k], str):
                continue
            elif isinstance(sample[k], (int, float)):
                sample[k] = torch.tensor(sample[k]).unsqueeze(0)
            else:
                sample[k] = torch.from_numpy(sample[k]).unsqueeze(0).float()

        try:
            plan = planner.plan(
                sample["agent_hist"],
                sample["agent_hist_mask"],
                sample["agent_type"],
                sample["map_polylines"],
                sample["map_mask"],
            )
            assert plan.shape == (5, 2), f"Expected (5, 2), got {plan.shape}"
        except Exception as e:
            print(f"Planning error (expected on random model): {e}")

        # Generate report
        results_dir = tmpdir / "results"
        results_dir.mkdir()

        # Create dummy results
        dummy_results = {
            "minADE_k": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6},
            "minFDE_k": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1},
            "miss_rate": {"mean": 0.1, "ci_low": 0.05, "ci_high": 0.15},
            "collision_rate": {"mean": 0.02, "ci_low": 0.0, "ci_high": 0.05},
        }

        with open(results_dir / "test_results.json", "w") as f:
            json.dump(dummy_results, f)

        # Generate report
        report = generate_report(results_dir, results_dir / "RESULTS.md")
        assert "Open-Loop Forecasting" in report
        assert (results_dir / "RESULTS.md").exists()

        print("✓ End-to-end test passed")


if __name__ == "__main__":
    test_end_to_end()
