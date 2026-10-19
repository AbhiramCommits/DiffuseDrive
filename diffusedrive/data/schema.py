"""Canonical data schema and validation."""

from typing import Any, Dict

import numpy as np
import torch


def validate_sample(sample: Dict[str, Any], T_hist: int = 11, T_fut: int = 80) -> None:
    """Validate sample has correct shapes and no NaNs."""
    required_keys = {
        "agent_hist": (None, T_hist, 5),
        "agent_hist_mask": (None, T_hist),
        "agent_fut": (None, T_fut, 2),
        "agent_fut_mask": (None, T_fut),
        "agent_type": (None,),
        "map_polylines": (None, None, 4),
        "map_mask": (None, None),
        "ego_index": (),
        "scene_id": (),
    }

    for key, expected_shape in required_keys.items():
        if key not in sample:
            raise ValueError(f"Missing key: {key}")

        val = sample[key]
        if isinstance(val, torch.Tensor):
            shape = tuple(val.shape)
            data = val
        else:
            val = np.asarray(val)
            shape = val.shape
            data = val

        # Check shape (None dims can be anything)
        if expected_shape:
            if len(shape) != len(expected_shape):
                raise ValueError(f"Shape mismatch for {key}: expected {len(expected_shape)} dims, got {len(shape)}")
            for i, (s, e) in enumerate(zip(shape, expected_shape)):
                if e is not None and s != e:
                    raise ValueError(f"Shape mismatch for {key} dim {i}: expected {e}, got {s}")

        # Check NaN for numeric types
        if key not in ("scene_id", "ego_index"):
            if np.isnan(data).any():
                raise ValueError(f"NaN found in {key}")


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Pad and batch scenes to [B, A, ...] and [B, P, S, ...] with masks."""
    B = len(batch)

    # Find max agents and polylines across batch
    max_agents = max(s["agent_hist"].shape[0] for s in batch)
    max_polylines = max(s["map_polylines"].shape[0] for s in batch)
    max_polyline_points = max(s["map_polylines"].shape[1] for s in batch)

    T_hist = batch[0]["agent_hist"].shape[1]
    T_fut = batch[0]["agent_fut"].shape[1]

    # Initialize padded tensors
    agent_hist = torch.zeros(B, max_agents, T_hist, 5)
    agent_hist_mask = torch.zeros(B, max_agents, T_hist, dtype=torch.bool)
    agent_fut = torch.zeros(B, max_agents, T_fut, 2)
    agent_fut_mask = torch.zeros(B, max_agents, T_fut, dtype=torch.bool)
    agent_type = torch.zeros(B, max_agents, dtype=torch.long)
    map_polylines = torch.zeros(B, max_polylines, max_polyline_points, 4)
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


def normalize_scene(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize to ego frame at t=0: translate + rotate."""
    # Get ego position and heading at t=0
    ego_idx = sample["ego_index"]
    ego_pos = sample["agent_hist"][ego_idx, -1, :2]  # last history frame
    ego_heading = np.arctan2(sample["agent_hist"][ego_idx, -1, 2],
                              sample["agent_hist"][ego_idx, -1, 3])

    # Rotation matrix
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    R = np.array([[c, s], [-s, c]])

    # Apply to agent history positions
    hist_pos = sample["agent_hist"][:, :, :2]  # [A, T, 2]
    hist_pos = (hist_pos - ego_pos[None, None, :]) @ R.T
    sample["agent_hist"][:, :, :2] = hist_pos

    # Rotate headings
    cos_h = sample["agent_hist"][:, :, 2]
    sin_h = sample["agent_hist"][:, :, 3]
    headings = np.arctan2(sin_h, cos_h)
    headings = headings - ego_heading
    sample["agent_hist"][:, :, 2] = np.cos(headings)
    sample["agent_hist"][:, :, 3] = np.sin(headings)

    # Apply to agent futures
    fut_pos = sample["agent_fut"][:, :, :2]
    fut_pos = (fut_pos - ego_pos[None, None, :]) @ R.T
    sample["agent_fut"][:, :, :2] = fut_pos

    # Apply to map polylines
    poly_pos = sample["map_polylines"][:, :, :2]
    poly_pos = (poly_pos - ego_pos[None, None, :]) @ R.T
    sample["map_polylines"][:, :, :2] = poly_pos

    # Rotate polyline deltas
    poly_delta = sample["map_polylines"][:, :, 2:4]
    poly_delta = poly_delta @ R.T
    sample["map_polylines"][:, :, 2:4] = poly_delta

    # Store transform for inverse
    sample["_norm_pos"] = ego_pos
    sample["_norm_heading"] = ego_heading

    return sample


def denormalize_scene(sample: Dict[str, Any], pred: np.ndarray) -> np.ndarray:
    """Inverse transform predictions back to world frame. pred: [A, T, 2]"""
    ego_pos = sample["_norm_pos"]
    ego_heading = sample["_norm_heading"]

    c, s = np.cos(ego_heading), np.sin(ego_heading)
    R = np.array([[c, s], [-s, c]])

    # pred is [A, T, 2] in ego frame
    pred = pred @ R.T + ego_pos[None, None, :]
    return pred
