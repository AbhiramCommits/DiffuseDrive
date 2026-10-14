"""Synthetic scenario generator with lane geometry and IDM dynamics."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any
from .schema import validate_sample


class SynthDriveDataset(Dataset):
    """Deterministic procedural scenes with lane geometry and multimodal futures."""

    def __init__(
        self,
        num_scenes: int = 100,
        T_hist: int = 11,
        T_fut: int = 80,
        seed: int = 42,
        fps: int = 10,
    ):
        self.num_scenes = num_scenes
        self.T_hist = T_hist
        self.T_fut = T_fut
        self.seed = seed
        self.fps = fps
        self.dt = 1.0 / fps

        np.random.seed(seed)

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a deterministic scene."""
        rng = np.random.RandomState(self.seed + idx)

        # Choose scenario type
        scenario_type = rng.choice(["straight", "curve", "intersection", "merge"])

        if scenario_type == "straight":
            return self._gen_straight(rng, idx)
        elif scenario_type == "curve":
            return self._gen_curve(rng, idx)
        elif scenario_type == "intersection":
            return self._gen_intersection(rng, idx)
        else:
            return self._gen_merge(rng, idx)

    def _gen_straight(self, rng: np.random.RandomState, idx: int) -> Dict[str, Any]:
        """Straight lane scenario."""
        num_agents = rng.randint(4, 12)

        # Lane: x from 0 to 100, y = 0 (one lane)
        # Agents spaced along lane, following each other with IDM
        positions = np.zeros((num_agents, 2))
        positions[:, 0] = np.linspace(0, 80, num_agents)
        positions[:, 1] = 0

        velocities = rng.uniform(5, 15, num_agents)  # m/s
        headings = np.zeros(num_agents)  # all facing x direction

        # Simulate to collect history and future
        hist = []
        fut = []

        state = {
            "pos": positions.copy(),
            "vel": velocities.copy(),
            "heading": headings.copy(),
        }

        # Burn-in steps
        for _ in range(20):
            state = self._idm_step(state, rng)

        # Collect history
        for _ in range(self.T_hist):
            hist.append(state["pos"].copy())
            state = self._idm_step(state, rng)

        # Collect future (multimodal: slight variations)
        for _ in range(self.T_fut):
            fut.append(state["pos"].copy())
            state = self._idm_step(state, rng)

        hist = np.array(hist)  # [T_hist, A, 2]
        fut = np.array(fut)    # [T_fut, A, 2]

        # Build agent trajectories: history [A, T_hist, 5]
        agent_hist = np.zeros((num_agents, self.T_hist, 5))
        for t in range(self.T_hist):
            pos = hist[t]
            agent_hist[:, t, :2] = pos
            if t > 0:
                vel = (pos - hist[t-1]) / self.dt
                speed = np.linalg.norm(vel, axis=1)
                heading = np.arctan2(vel[:, 1] + 1e-6, vel[:, 0])
                agent_hist[:, t, 2] = np.cos(heading)
                agent_hist[:, t, 3] = np.sin(heading)
                agent_hist[:, t, 4] = speed
            else:
                agent_hist[:, t, 2] = 1  # cos(0)
                agent_hist[:, t, 3] = 0  # sin(0)
                agent_hist[:, t, 4] = 5

        # Future [A, T_fut, 2]
        agent_fut = fut.transpose(1, 0, 2)

        # Map: simple lane boundaries
        num_polylines = 2
        max_points = 20
        map_polylines = np.zeros((num_polylines, max_points, 4))
        map_mask = np.zeros((num_polylines, max_points), dtype=bool)

        # Left boundary
        xs = np.linspace(0, 100, max_points)
        map_polylines[0, :, 0] = xs
        map_polylines[0, :, 1] = -2  # 2m left
        map_polylines[0, 1:, 2] = np.diff(xs)
        map_polylines[0, :, 3] = 0
        map_mask[0] = True

        # Right boundary
        map_polylines[1, :, 0] = xs
        map_polylines[1, :, 1] = 2  # 2m right
        map_polylines[1, 1:, 2] = np.diff(xs)
        map_polylines[1, :, 3] = 0
        map_mask[1] = True

        # Random ego index
        ego_idx = rng.randint(0, num_agents)

        sample = {
            "agent_hist": agent_hist,
            "agent_hist_mask": np.ones((num_agents, self.T_hist), dtype=bool),
            "agent_fut": agent_fut,
            "agent_fut_mask": np.ones((num_agents, self.T_fut), dtype=bool),
            "agent_type": rng.choice([0, 0, 0, 1], num_agents),  # mostly vehicles
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "ego_index": ego_idx,
            "scene_id": f"synth_straight_{self.seed}_{idx:06d}",
        }

        validate_sample(sample, self.T_hist, self.T_fut)
        return sample

    def _gen_curve(self, rng: np.random.RandomState, idx: int) -> Dict[str, Any]:
        """Curved lane scenario."""
        num_agents = rng.randint(3, 8)

        # Circular arc lane
        radius = 50
        angles = np.linspace(0, np.pi / 3, num_agents)
        positions = np.zeros((num_agents, 2))
        positions[:, 0] = radius * np.sin(angles)
        positions[:, 1] = radius * (1 - np.cos(angles))

        velocities = rng.uniform(5, 12, num_agents)
        headings = angles.copy()

        hist = []
        fut = []

        state = {"pos": positions.copy(), "vel": velocities.copy(), "heading": headings.copy()}

        for _ in range(20):
            state = self._curve_step(state, radius, rng)

        for _ in range(self.T_hist):
            hist.append(state["pos"].copy())
            state = self._curve_step(state, radius, rng)

        for _ in range(self.T_fut):
            fut.append(state["pos"].copy())
            state = self._curve_step(state, radius, rng)

        hist = np.array(hist)
        fut = np.array(fut)

        agent_hist = np.zeros((num_agents, self.T_hist, 5))
        for t in range(self.T_hist):
            pos = hist[t]
            agent_hist[:, t, :2] = pos
            if t > 0:
                vel = (pos - hist[t-1]) / self.dt
                speed = np.linalg.norm(vel, axis=1)
                heading = np.arctan2(vel[:, 1] + 1e-6, vel[:, 0])
                agent_hist[:, t, 2] = np.cos(heading)
                agent_hist[:, t, 3] = np.sin(heading)
                agent_hist[:, t, 4] = speed
            else:
                agent_hist[:, t, 2] = np.cos(headings)
                agent_hist[:, t, 3] = np.sin(headings)
                agent_hist[:, t, 4] = 8

        agent_fut = fut.transpose(1, 0, 2)

        # Map: curved lanes
        num_polylines = 2
        max_points = 25
        map_polylines = np.zeros((num_polylines, max_points, 4))
        map_mask = np.zeros((num_polylines, max_points), dtype=bool)

        thetas = np.linspace(0, np.pi / 2, max_points)
        for i in range(2):
            r = radius + (i - 0.5) * 3
            xs = r * np.sin(thetas)
            ys = r * (1 - np.cos(thetas))
            map_polylines[i, :, 0] = xs
            map_polylines[i, :, 1] = ys
            map_polylines[i, 1:, 2] = np.diff(xs)
            map_polylines[i, 1:, 3] = np.diff(ys)
            map_mask[i] = True

        ego_idx = rng.randint(0, num_agents)

        sample = {
            "agent_hist": agent_hist,
            "agent_hist_mask": np.ones((num_agents, self.T_hist), dtype=bool),
            "agent_fut": agent_fut,
            "agent_fut_mask": np.ones((num_agents, self.T_fut), dtype=bool),
            "agent_type": rng.choice([0, 0, 1], num_agents),
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "ego_index": ego_idx,
            "scene_id": f"synth_curve_{self.seed}_{idx:06d}",
        }

        validate_sample(sample, self.T_hist, self.T_fut)
        return sample

    def _gen_intersection(self, rng: np.random.RandomState, idx: int) -> Dict[str, Any]:
        """4-way intersection with multimodal futures."""
        # 4 lanes converging at (0, 0)
        # East-West: y=0, x from -50 to 50
        # North-South: x=0, y from -50 to 50

        num_agents = rng.randint(6, 16)
        positions = np.zeros((num_agents, 2))
        velocities = np.zeros(num_agents)
        headings = np.zeros(num_agents)

        # Distribute agents on 4 approaches
        for i in range(num_agents):
            approach = i % 4
            dist_to_intersection = rng.uniform(10, 40)

            if approach == 0:  # Westbound
                positions[i] = [dist_to_intersection, 0]
                headings[i] = np.pi
                velocities[i] = rng.uniform(5, 15)
            elif approach == 1:  # Eastbound
                positions[i] = [-dist_to_intersection, 0]
                headings[i] = 0
                velocities[i] = rng.uniform(5, 15)
            elif approach == 2:  # Northbound
                positions[i] = [0, -dist_to_intersection]
                headings[i] = np.pi / 2
                velocities[i] = rng.uniform(5, 15)
            else:  # Southbound
                positions[i] = [0, dist_to_intersection]
                headings[i] = -np.pi / 2
                velocities[i] = rng.uniform(5, 15)

        hist = []
        fut = []
        state = {"pos": positions.copy(), "vel": velocities.copy(), "heading": headings.copy()}

        for _ in range(20):
            state = self._intersection_step(state, rng)

        for _ in range(self.T_hist):
            hist.append(state["pos"].copy())
            state = self._intersection_step(state, rng)

        # Multimodal: agents at intersection can turn
        for _ in range(self.T_fut):
            fut.append(state["pos"].copy())
            state = self._intersection_step(state, rng)

        hist = np.array(hist)
        fut = np.array(fut)

        agent_hist = np.zeros((num_agents, self.T_hist, 5))
        for t in range(self.T_hist):
            pos = hist[t]
            agent_hist[:, t, :2] = pos
            if t > 0:
                vel = (pos - hist[t-1]) / self.dt
                speed = np.linalg.norm(vel, axis=1)
                heading = np.arctan2(vel[:, 1] + 1e-6, vel[:, 0])
                agent_hist[:, t, 2] = np.cos(heading)
                agent_hist[:, t, 3] = np.sin(heading)
                agent_hist[:, t, 4] = speed
            else:
                agent_hist[:, t, 2] = np.cos(headings)
                agent_hist[:, t, 3] = np.sin(headings)
                agent_hist[:, t, 4] = velocities

        agent_fut = fut.transpose(1, 0, 2)

        # Map: 4 lanes
        num_polylines = 8
        max_points = 20
        map_polylines = np.zeros((num_polylines, max_points, 4))
        map_mask = np.zeros((num_polylines, max_points), dtype=bool)

        # Horizontal lanes (y = ±2, ±6)
        for lane_id in range(2):
            y = (-1) ** lane_id * (lane_id + 1) * 3
            xs = np.linspace(-50, 50, max_points)
            map_polylines[lane_id, :, 0] = xs
            map_polylines[lane_id, :, 1] = y
            map_polylines[lane_id, 1:, 2] = np.diff(xs)
            map_polylines[lane_id, :, 3] = 0
            map_mask[lane_id] = True

        # Vertical lanes (x = ±2, ±6)
        for lane_id in range(2):
            x = (-1) ** lane_id * (lane_id + 1) * 3
            ys = np.linspace(-50, 50, max_points)
            map_polylines[2 + lane_id, :, 0] = x
            map_polylines[2 + lane_id, :, 1] = ys
            map_polylines[2 + lane_id, :, 2] = 0
            map_polylines[2 + lane_id, 1:, 3] = np.diff(ys)
            map_mask[2 + lane_id] = True

        ego_idx = rng.randint(0, num_agents)

        sample = {
            "agent_hist": agent_hist,
            "agent_hist_mask": np.ones((num_agents, self.T_hist), dtype=bool),
            "agent_fut": agent_fut,
            "agent_fut_mask": np.ones((num_agents, self.T_fut), dtype=bool),
            "agent_type": rng.choice([0, 0, 1], num_agents),
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "ego_index": ego_idx,
            "scene_id": f"synth_intersection_{self.seed}_{idx:06d}",
        }

        validate_sample(sample, self.T_hist, self.T_fut)
        return sample

    def _gen_merge(self, rng: np.random.RandomState, idx: int) -> Dict[str, Any]:
        """Lane merge scenario."""
        num_agents = rng.randint(5, 12)

        # Main lane (y=0) and merging lane (y=2)
        positions = np.zeros((num_agents, 2))
        velocities = np.zeros(num_agents)
        headings = np.zeros(num_agents)

        # Some agents on main lane, some on merging lane
        for i in range(num_agents):
            if i < num_agents // 2:
                positions[i, 0] = rng.uniform(0, 80)
                positions[i, 1] = 0
            else:
                positions[i, 0] = rng.uniform(10, 80)
                positions[i, 1] = 2
            headings[i] = 0
            velocities[i] = rng.uniform(6, 14)

        hist = []
        fut = []
        state = {"pos": positions.copy(), "vel": velocities.copy(), "heading": headings.copy()}

        for _ in range(20):
            state = self._merge_step(state, rng)

        for _ in range(self.T_hist):
            hist.append(state["pos"].copy())
            state = self._merge_step(state, rng)

        for _ in range(self.T_fut):
            fut.append(state["pos"].copy())
            state = self._merge_step(state, rng)

        hist = np.array(hist)
        fut = np.array(fut)

        agent_hist = np.zeros((num_agents, self.T_hist, 5))
        for t in range(self.T_hist):
            pos = hist[t]
            agent_hist[:, t, :2] = pos
            if t > 0:
                vel = (pos - hist[t-1]) / self.dt
                speed = np.linalg.norm(vel, axis=1)
                heading = np.arctan2(vel[:, 1] + 1e-6, vel[:, 0])
                agent_hist[:, t, 2] = np.cos(heading)
                agent_hist[:, t, 3] = np.sin(heading)
                agent_hist[:, t, 4] = speed
            else:
                agent_hist[:, t, 2] = 1
                agent_hist[:, t, 3] = 0
                agent_hist[:, t, 4] = velocities

        agent_fut = fut.transpose(1, 0, 2)

        # Map: 2 lanes
        num_polylines = 4
        max_points = 20
        map_polylines = np.zeros((num_polylines, max_points, 4))
        map_mask = np.zeros((num_polylines, max_points), dtype=bool)

        xs = np.linspace(0, 100, max_points)
        for i in range(2):
            y = i * 3 - 1.5
            map_polylines[i, :, 0] = xs
            map_polylines[i, :, 1] = y
            map_polylines[i, 1:, 2] = np.diff(xs)
            map_polylines[i, :, 3] = 0
            map_mask[i] = True

        ego_idx = rng.randint(0, num_agents)

        sample = {
            "agent_hist": agent_hist,
            "agent_hist_mask": np.ones((num_agents, self.T_hist), dtype=bool),
            "agent_fut": agent_fut,
            "agent_fut_mask": np.ones((num_agents, self.T_fut), dtype=bool),
            "agent_type": rng.choice([0, 0, 1], num_agents),
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "ego_index": ego_idx,
            "scene_id": f"synth_merge_{self.seed}_{idx:06d}",
        }

        validate_sample(sample, self.T_hist, self.T_fut)
        return sample

    def _idm_step(self, state: Dict, rng: np.random.RandomState) -> Dict:
        """IDM car following dynamics (1D, straight lane)."""
        pos = state["pos"]
        vel = state["vel"]

        # Simple 1D IDM on x-axis with lane y-offset
        a0 = 2  # desired accel
        v_des = rng.normal(10, 2, vel.shape)
        v_des = np.clip(v_des, 3, 20)
        d0 = 2  # safe distance
        tau = 1

        # Distance to front vehicle
        x_sorted = np.argsort(pos[:, 0])
        accel = np.zeros_like(vel)

        for i, idx in enumerate(x_sorted):
            if i < len(x_sorted) - 1:
                next_idx = x_sorted[i + 1]
                dx = pos[next_idx, 0] - pos[idx, 0]
                dv = vel[idx] - vel[next_idx]
                s_star = d0 + vel[idx] * tau + vel[idx] * dv / (2 * np.sqrt(a0 * 2))
                accel[idx] = a0 * (1 - (vel[idx] / (v_des[idx] + 1e-6)) ** 2 - (s_star / (dx + 1)) ** 2)
            else:
                accel[idx] = a0 * (1 - (vel[idx] / (v_des[idx] + 1e-6)) ** 2)

        accel = np.clip(accel, -3, 3)
        vel = np.clip(vel + accel * self.dt, 0.5, 20)
        pos = pos + np.array([vel, np.zeros_like(vel)]).T * self.dt

        return {"pos": pos, "vel": vel, "heading": state["heading"]}

    def _curve_step(self, state: Dict, radius: float, rng: np.random.RandomState) -> Dict:
        """Follow curved path."""
        pos = state["pos"]
        vel = state["vel"]
        heading = state["heading"]

        # Move along curve
        speed = np.clip(vel + rng.normal(0, 0.1, vel.shape), 3, 15)
        angle_vel = speed / radius

        heading += angle_vel * self.dt
        pos[:, 0] = radius * np.sin(heading)
        pos[:, 1] = radius * (1 - np.cos(heading))

        return {"pos": pos, "vel": speed, "heading": heading}

    def _intersection_step(self, state: Dict, rng: np.random.RandomState) -> Dict:
        """Navigate intersection with random turns."""
        pos = state["pos"]
        vel = state["vel"]
        heading = state["heading"]

        # Agents within 5m of center may turn randomly
        dist_to_center = np.linalg.norm(pos, axis=1)
        turn_mask = dist_to_center < 5

        for i in np.where(turn_mask)[0]:
            if rng.rand() < 0.05:
                heading[i] += rng.choice([-np.pi / 2, np.pi / 2])

        vel = np.clip(vel + rng.normal(0, 0.2, vel.shape), 2, 15)
        pos = pos + np.array([vel * np.cos(heading), vel * np.sin(heading)]).T * self.dt

        return {"pos": pos, "vel": vel, "heading": heading}

    def _merge_step(self, state: Dict, rng: np.random.RandomState) -> Dict:
        """Handle merging with lane changes."""
        pos = state["pos"]
        vel = state["vel"]
        heading = state["heading"]

        # Vehicles on right lane may drift left
        for i in range(len(pos)):
            if pos[i, 1] > 1:
                pos[i, 1] = np.clip(pos[i, 1] + rng.normal(-0.3, 0.2), 0, 2)

        vel = np.clip(vel + rng.normal(0, 0.15, vel.shape), 3, 16)
        pos = pos + np.array([vel, np.zeros_like(vel)]).T * self.dt

        return {"pos": pos, "vel": vel, "heading": heading}
