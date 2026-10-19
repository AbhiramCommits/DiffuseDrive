"""Batching and collation utilities."""

from typing import Any, Dict, List

import torch


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate samples into padded batch tensors."""
    B = len(batch)

    # Find max dimensions
    max_agents = max(s["agent_hist"].shape[0] for s in batch)
    max_polylines = max(s["map_polylines"].shape[0] for s in batch)
    max_polyline_points = max(s["map_polylines"].shape[1] for s in batch)

    T_hist = batch[0]["agent_hist"].shape[1]
    T_fut = batch[0]["agent_fut"].shape[1]

    # Initialize padded tensors
    agent_hist = torch.zeros(B, max_agents, T_hist, 5, dtype=torch.float32)
    agent_hist_mask = torch.zeros(B, max_agents, T_hist, dtype=torch.bool)
    agent_fut = torch.zeros(B, max_agents, T_fut, 2, dtype=torch.float32)
    agent_fut_mask = torch.zeros(B, max_agents, T_fut, dtype=torch.bool)
    agent_type = torch.zeros(B, max_agents, dtype=torch.long)
    map_polylines = torch.zeros(B, max_polylines, max_polyline_points, 4, dtype=torch.float32)
    map_mask = torch.zeros(B, max_polylines, max_polyline_points, dtype=torch.bool)
    ego_index = torch.zeros(B, dtype=torch.long)
    scene_ids = []

    for b, sample in enumerate(batch):
        A = sample["agent_hist"].shape[0]
        P = sample["map_polylines"].shape[0]
        S = sample["map_polylines"].shape[1]

        agent_hist[b, :A] = torch.as_tensor(sample["agent_hist"], dtype=torch.float32)
        agent_hist_mask[b, :A] = torch.as_tensor(sample["agent_hist_mask"], dtype=torch.bool)
        agent_fut[b, :A] = torch.as_tensor(sample["agent_fut"], dtype=torch.float32)
        agent_fut_mask[b, :A] = torch.as_tensor(sample["agent_fut_mask"], dtype=torch.bool)
        agent_type[b, :A] = torch.as_tensor(sample["agent_type"], dtype=torch.long)
        map_polylines[b, :P, :S] = torch.as_tensor(sample["map_polylines"], dtype=torch.float32)
        map_mask[b, :P, :S] = torch.as_tensor(sample["map_mask"], dtype=torch.bool)
        ego_index[b] = sample["ego_index"]
        scene_ids.append(sample["scene_id"])

    return {
        "agent_hist": agent_hist,
        "agent_hist_mask": agent_hist_mask,
        "agent_fut": agent_fut,
        "agent_fut_mask": agent_fut_mask,
        "agent_type": agent_type,
        "map_polylines": map_polylines,
        "map_mask": map_mask,
        "ego_index": ego_index,
        "scene_ids": scene_ids,
    }
