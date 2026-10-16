"""Evaluation CLI."""

import argparse
import json
from pathlib import Path

from .harness import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--config", default="configs/small_cpu.yaml", help="Config path")
    parser.add_argument("--K", type=int, default=6, help="Number of samples")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    results = evaluate(
        args.ckpt,
        args.config,
        K=args.K,
        num_steps=args.steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {out_path}")
    for metric, vals in results.items():
        print(f"  {metric}: {vals['mean']:.4f} [{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
