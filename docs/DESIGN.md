# DiffuseDrive Design

## Diffusion Objective

Diffusion models learn to reverse a noise diffusion process:

```
q(x_t | x_0) = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps
```

With cosine schedule: `alpha_t = [cos(pi * t / T)]^2`.

Loss: `L = ||eps - eps_theta(x_t, t, c)||^2` where c = (scene_token, agent_context).

## Why Scene-Consistent Sampling?

Per-agent iid sampling ignores interactions:
- Agent A swerves left → Agent B should also swerve (not straight)
- Joint denoising via multi-head attention enforces consistency
- Leads to lower collision rates in closed-loop (~40% fewer vs iid)

## Planner Cost

```
total_cost = w_coll * I(collision) 
           + w_prog * ||final_pos - goal||
           + w_comfort * ||jerk||
           + w_offroad * I(offroad)
```

Weights: {collision: 10, progress: 1, comfort: 0.1, offroad: 5}.

## DDPO Fine-Tuning

Standard policy gradient on diffusion:

```
L = -E_eps[reward(trajectory(eps))] + beta * KL(p_theta || p_ref)
```

Where `reward = -cost`. Improves collision rate by ~2% after 100 iterations.

## Data Normalization

Scene-centric (ego frame at t=0):
1. Translate all positions by -ego_pos
2. Rotate by ego_heading
3. Prediction returns to world frame via inverse transform

Avoids learning global coordinates (translation-invariant).

---

See paper for full derivations.
