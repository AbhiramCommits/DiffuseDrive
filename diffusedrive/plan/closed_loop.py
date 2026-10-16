"""Closed-loop planning evaluation."""

import numpy as np

from diffusedrive.data.collate import collate_batch


def run_episodes(
    planner,
    dataset,
    n_episodes: int = 100,
    seed: int = 0,
) -> dict:
    """Run closed-loop episodes. Return metrics."""
    np.random.seed(seed)

    collision_count = 0
    offroad_count = 0
    progress_sum = 0.0
    jerk_sum = 0.0

    for i in range(min(n_episodes, len(dataset))):
        # Batch of one, with the dtypes the models expect (long agent_type, bool masks)
        sample = collate_batch([dataset[i]])

        ego_idx = int(sample["ego_index"][0])
        gt_fut = sample["agent_fut"].numpy()[0]  # [A, T_fut, 2] (all agents)

        # Plan
        try:
            plan = planner.plan(
                sample["agent_hist"],
                sample["agent_hist_mask"],
                sample["agent_type"],
                sample["map_polylines"],
                sample["map_mask"],
                ego_idx=ego_idx,
            )
        except Exception:
            collision_count += 1
            continue

        # Evaluate
        # Collision: check against GT other agents
        for t in range(min(len(plan), gt_fut.shape[1])):
            for a in range(gt_fut.shape[0]):
                if a != ego_idx:
                    other_pos = gt_fut[a, t, :2]
                    if np.linalg.norm(plan[t] - other_pos) < 1.0:
                        collision_count += 1
                        break

        # Progress: forward distance
        progress = np.linalg.norm(plan[-1] - plan[0])
        progress_sum += progress

        # Jerk
        if len(plan) > 3:
            vel = np.diff(plan, axis=0)
            accel = np.diff(vel, axis=0)
            jerk = np.diff(accel, axis=0)
            jerk_sum += np.linalg.norm(jerk) / max(1.0, len(jerk))

        # Offroad
        for pos in plan:
            if pos[0] < -50 or pos[0] > 50 or pos[1] < -50 or pos[1] > 50:
                offroad_count += 1
                break

    return {
        "collision_rate": collision_count / max(1, n_episodes),
        "offroad_rate": offroad_count / max(1, n_episodes),
        "mean_progress": progress_sum / max(1, n_episodes),
        "mean_jerk": jerk_sum / max(1, n_episodes),
    }
