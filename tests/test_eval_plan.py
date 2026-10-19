"""Unit tests for metrics, planning costs, planners, the fallback simulator and helpers."""

import numpy as np
import pytest
import torch

from diffusedrive.config import load_config
from diffusedrive.data import build_dataset, collate_batch
from diffusedrive.eval.metrics import (
    bootstrap_ci,
    collision_rate,
    compute_metrics,
    jointADE,
    minADE_k,
    minFDE_k,
    miss_rate,
    nll_kde,
)
from diffusedrive.models import build_model
from diffusedrive.plan.closed_loop import run_episodes
from diffusedrive.plan.cost import (
    collision_cost,
    comfort_cost,
    offroad_cost,
    progress_cost,
    total_cost,
)
from diffusedrive.plan.planner import BCPlanner, ConstantVelocityPlanner, SamplingPlanner
from diffusedrive.sim import BicycleModel, check_collision, check_offroad
from diffusedrive.utils.seed import set_seed
from diffusedrive.viz import plot_scene

# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _straight_gt(A=2, T=5):
    t = np.arange(1, T + 1, dtype=float)
    gt = np.zeros((A, T, 2))
    for a in range(A):
        gt[a, :, 0] = t
        gt[a, :, 1] = 10.0 * a
    return gt


def test_min_ade_fde_pick_the_best_sample():
    gt = _straight_gt()
    good = gt.copy()
    bad = gt + np.array([3.0, 4.0])  # constant 5 m offset
    pred = np.stack([bad, good, bad])  # [K=3, A, T, 2]

    assert minADE_k(pred, gt, k=3) == pytest.approx(0.0)
    assert minFDE_k(pred, gt, k=3) == pytest.approx(0.0)
    # only the first (bad) sample is considered with k=1
    assert minADE_k(pred, gt, k=1) == pytest.approx(5.0)
    assert minFDE_k(pred, gt, k=1) == pytest.approx(5.0)


def test_min_ade_is_mean_of_pointwise_errors():
    gt = _straight_gt(A=1, T=4)
    pred = gt.copy()[None]
    pred[0, 0, -1] += np.array([4.0, 0.0])  # 4 m error on the last step only
    assert minADE_k(pred, gt) == pytest.approx(1.0)
    assert minFDE_k(pred, gt) == pytest.approx(4.0)


def test_miss_rate_uses_threshold():
    gt = _straight_gt(A=2)
    pred = gt.copy()[None]
    pred[0, 1, -1] += np.array([0.0, 3.0])  # agent 1 misses by 3 m
    assert miss_rate(pred, gt, threshold=2.0) == pytest.approx(0.5)
    assert miss_rate(pred, gt, threshold=5.0) == pytest.approx(0.0)


def test_joint_ade_zero_for_perfect_prediction():
    gt = _straight_gt()
    assert jointADE(gt[None], gt) == pytest.approx(0.0)


def test_collision_rate_detects_overlap():
    far = _straight_gt(A=2)  # agents 10 m apart
    assert collision_rate(far[None], radius=0.5) == 0.0
    close = far.copy()
    close[1, :, 1] = 0.2  # agents 0.2 m apart at every step
    assert collision_rate(close[None], radius=0.5) == 1.0


def test_nll_kde_finite_and_lower_for_matching_samples():
    rng = np.random.default_rng(0)
    gt = _straight_gt()
    near = gt[None] + rng.normal(0, 0.1, size=(6,) + gt.shape)
    far = near + 50.0
    assert np.isfinite(nll_kde(near, gt))
    assert nll_kde(near, gt) < nll_kde(far, gt)


def test_nll_kde_degenerate_samples_returns_inf():
    gt = _straight_gt()
    assert nll_kde(np.zeros((2, 2, 5, 2)), gt) == np.inf


def test_compute_metrics_keys_and_collision_radius():
    gt = _straight_gt(A=2)
    m = compute_metrics(gt[None].repeat(6, axis=0), gt, K=6)
    assert set(m) == {"minADE_k", "minFDE_k", "miss_rate", "jointADE", "collision_rate", "nll_kde"}
    assert m["minADE_k"] == pytest.approx(0.0)
    # agents are 10 m apart; K must not leak into the collision radius
    assert m["collision_rate"] == 0.0


def test_bootstrap_ci_brackets_the_mean():
    scores = np.arange(100, dtype=float)
    mean, low, high = bootstrap_ci(scores, n_bootstrap=200)
    assert low <= mean <= high
    assert mean == pytest.approx(49.5, abs=3.0)
    assert bootstrap_ci(scores, n_bootstrap=200) == (mean, low, high)  # seeded


# --------------------------------------------------------------------------- #
# planning costs + simulator
# --------------------------------------------------------------------------- #


def test_costs():
    ego = np.stack([np.arange(5.0), np.zeros(5)], axis=1)
    others_far = ego[None] + np.array([0.0, 10.0])
    others_hit = ego[None] + np.array([0.0, 0.5])
    goal = np.array([4.0, 0.0])
    bounds = (-10, 10, -10, 10)

    assert collision_cost(ego, others_far) == 0.0
    assert collision_cost(ego, others_hit) == 1.0
    assert progress_cost(ego, goal) == pytest.approx(0.0)
    assert progress_cost(ego, np.array([7.0, 4.0])) == pytest.approx(5.0)
    assert comfort_cost(ego) == pytest.approx(0.0)  # constant velocity -> no jerk
    assert comfort_cost(ego[:3]) == 0.0
    assert offroad_cost(ego, bounds) == 0.0
    assert offroad_cost(ego, (-1, 2, -1, 1)) == 1.0
    assert total_cost(ego, others_far, goal, bounds) == pytest.approx(0.0)
    assert total_cost(ego, others_hit, goal, bounds) == pytest.approx(10.0)
    w = {"collision": 1.0, "progress": 0.0, "comfort": 0.0, "offroad": 0.0}
    assert total_cost(ego, others_hit, goal, bounds, weights=w) == pytest.approx(1.0)


def test_bicycle_model_and_checks():
    sim = BicycleModel(length=0.5, dt=0.1)
    state = np.array([0.0, 0.0, 0.0, 1.0])
    nxt = sim.step(state, np.array([0.0, 0.0]))
    assert nxt == pytest.approx([0.1, 0.0, 0.0, 1.0])
    turned = sim.step(state, np.array([10.0, 0.3]))
    assert turned[3] == pytest.approx(2.0) and turned[2] > 0
    assert check_collision(np.zeros(2), np.array([0.5, 0.0]))
    assert not check_collision(np.zeros(2), np.array([2.0, 0.0]))
    assert not check_offroad(np.zeros(2), (-1, 1, -1, 1))
    assert check_offroad(np.array([2.0, 0.0]), (-1, 1, -1, 1))


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (np.random.rand(), torch.rand(1).item())
    set_seed(123, deterministic=False)
    b = (np.random.rand(), torch.rand(1).item())
    assert a == b


# --------------------------------------------------------------------------- #
# planners + closed loop (tiny models, short horizon)
# --------------------------------------------------------------------------- #


def _tiny(model_name: str, T_fut: int = 8):
    cfg = load_config("configs/small_cpu.yaml")
    cfg.data.num_train_scenes = 3
    cfg.data.T_fut = T_fut
    cfg.model.model = model_name
    cfg.model.diffusion_steps = 10
    torch.manual_seed(0)
    return cfg, build_dataset(cfg.data), build_model(cfg).eval()


def _plan_args(batch):
    return (
        batch["agent_hist"],
        batch["agent_hist_mask"],
        batch["agent_type"],
        batch["map_polylines"],
        batch["map_mask"],
    )


def test_constant_velocity_planner_extrapolates():
    hist = torch.zeros(1, 1, 3, 5)
    hist[0, 0, :, 0] = torch.tensor([0.0, 0.1, 0.2])  # 1 m/s along x
    plan = ConstantVelocityPlanner(planning_horizon=4).plan(hist, None, None, None, None)
    assert plan.shape == (4, 2)
    assert plan[:, 0] == pytest.approx([0.3, 0.4, 0.5, 0.6])
    assert plan[:, 1] == pytest.approx([0.0] * 4)


def test_sampling_and_bc_planners_return_horizon():
    _, ds, diffusion = _tiny("diffusion")
    batch = collate_batch([ds[0]])
    plan = SamplingPlanner(diffusion, planning_horizon=5).plan(*_plan_args(batch), K=2)
    assert plan.shape == (5, 2) and np.isfinite(plan).all()

    _, _, bc = _tiny("bc")
    plan = BCPlanner(bc, planning_horizon=5).plan(*_plan_args(batch))
    assert plan.shape == (5, 2) and np.isfinite(plan).all()


def test_run_episodes_closed_loop():
    _, ds, _ = _tiny("bc")
    metrics = run_episodes(ConstantVelocityPlanner(planning_horizon=5), ds, n_episodes=3)
    assert set(metrics) == {"collision_rate", "offroad_rate", "mean_progress", "mean_jerk"}
    assert 0.0 <= metrics["collision_rate"] <= 1.0
    assert 0.0 <= metrics["offroad_rate"] <= 1.0
    assert metrics["mean_progress"] >= 0.0


def test_plot_scene_writes_png(tmp_path):
    gt = _straight_gt(A=2)
    poly = np.full((2, 4, 4), np.nan)
    poly[0, :, :2] = [[0, -1], [1, -1], [2, -1], [3, -1]]
    out = tmp_path / "fig" / "scene.png"
    plot_scene(gt[None].repeat(2, axis=0), gt, poly, out)
    assert out.exists() and out.stat().st_size > 0


def test_evaluate_harness_end_to_end(tmp_path):
    import yaml

    from diffusedrive.eval.harness import evaluate

    raw = yaml.safe_load(open("configs/small_cpu.yaml"))
    raw["data"].update(num_train_scenes=4, T_fut=8)
    raw["model"]["diffusion_steps"] = 10
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    cfg = load_config(cfg_path)
    model = build_model(cfg)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)

    result = evaluate(str(ckpt), str(cfg_path), K=2, num_steps=2)
    assert set(result) == {"minADE_k", "minFDE_k", "miss_rate", "collision_rate"}
    for stats in result.values():
        assert np.isfinite(stats["mean"])
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
