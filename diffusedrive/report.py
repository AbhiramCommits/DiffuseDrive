"""Report generator: reads results and emits markdown."""

import json
from pathlib import Path
from typing import Dict, List


def generate_report(results_dir: Path = Path("results"), output_path: Path = Path("results/RESULTS.md")):
    """Generate markdown report from JSON results."""

    # Load all results
    eval_results = {}
    planning_results = {}

    for json_path in results_dir.glob("*.json"):
        with open(json_path) as f:
            data = json.load(f)
            if "diffusion" in json_path.name or "mixture" in json_path.name or "bc" in json_path.name:
                eval_results[json_path.stem] = data
            elif "closed_loop" in json_path.name:
                planning_results[json_path.stem] = data

    # Build markdown
    md = """# DiffuseDrive: Results

## Open-Loop Forecasting

| Model | K | minADE | minFDE | Miss Rate | Collision |
|-------|---|--------|--------|-----------|-----------|
"""

    for name, results in sorted(eval_results.items()):
        if "minADE_k" in results:
            md += f"| {name.replace('_', ' ')} | - | "
            md += f"{results['minADE_k']['mean']:.3f} | "
            md += f"{results['minFDE_k']['mean']:.3f} | "
            md += f"{results['miss_rate']['mean']:.3f} | "
            md += f"{results['collision_rate']['mean']:.3f} |\n"

    md += """
## Closed-Loop Planning

| Planner | Collision Rate | Progress | Jerk RMS |
|---------|----------------|----------|----------|
"""

    for name, results in sorted(planning_results.items()):
        md += f"| {name.replace('_', ' ')} | "
        md += f"{results.get('collision_rate', 0):.3f} | "
        md += f"{results.get('mean_progress', 0):.3f} | "
        md += f"{results.get('mean_jerk', 0):.3f} |\n"

    md += """
## Architecture

- **Data**: SynthDrive procedural scenarios (straight, curve, intersection, merge)
- **Encoder**: PolylineEncoder + AgentHistoryEncoder + agent-agent/agent-map attention
- **Diffusion**: Cosine schedule, DDPM/DDIM, FiLM conditioning
- **Planner**: K-sample scoring with collision/progress/comfort/offroad costs
- **RL**: DDPO-style reward-guided fine-tuning

## Reproducibility

All results deterministic per seed (42). Metrics computed with 1000-sample bootstrap 95% CI.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)

    print(f"✓ Report saved to {output_path}")
    return md


if __name__ == "__main__":
    generate_report()
