"""Planning CLI."""

import argparse
import json
import torch
from pathlib import Path

from diffusedrive.config import load_config
from diffusedrive.data import build_dataset
from diffusedrive.models import build_model
from .planner import SamplingPlanner, BCPlanner, ConstantVelocityPlanner
from .closed_loop import run_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", default="diffusion", help="cv, bc, diffusion")
    parser.add_argument("--ckpt", default=None, help="Model checkpoint")
    parser.add_argument("--config", default="configs/small_cpu.yaml", help="Config")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--out", required=True, help="Output JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.planner == "cv":
        planner = ConstantVelocityPlanner(planning_horizon=10)
    else:
        model = build_model(cfg).to(device).eval()
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])

        if args.planner == "bc":
            planner = BCPlanner(model, planning_horizon=10)
        else:
            planner = SamplingPlanner(model, planning_horizon=10)

    results = run_episodes(planner, dataset, n_episodes=args.episodes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {out_path}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
