# Contributing

## Setup

```bash
git clone https://github.com/...
cd DiffuseDrive
pip install -e .[dev]
```

## Running Tests

```bash
make test          # Full suite
pytest tests/test_models.py -v
make reproduce    # Full pipeline
```

## Adding a New Model

1. Implement `forward(agent_hist, ..., agent_fut) -> loss` in `models/`.
2. Add to `build_model()` dispatcher in `models/__init__.py`.
3. Add test in `tests/test_models.py`.
4. Train: `dd-train --model <name> --out runs/<name>`

## Adding a New Metric

1. Implement in `eval/metrics.py`.
2. Call in `eval/harness.py` within `compute_metrics()`.
3. Update report template in `diffusedrive/report.py`.

## Code Style

```bash
ruff check .  # Lint
```

Target: 70%+ coverage, all tests passing.

## Publishing

Requires write access to `main` branch.

1. Create feature branch: `git checkout -b feature/xyz`
2. Make changes, test locally: `make test`
3. Push and open PR
4. CI must pass
5. Merge to main
