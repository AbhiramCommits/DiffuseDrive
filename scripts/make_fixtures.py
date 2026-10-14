#!/usr/bin/env python
"""Generate test fixtures for unit tests."""

import os
import numpy as np
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusedrive.data.synth import SynthDriveDataset
from diffusedrive.data.schema import validate_sample


def main():
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Generate 4 synthetic scenes
    dataset = SynthDriveDataset(num_scenes=4, seed=0)

    for i in range(4):
        scene = dataset[i]
        validate_sample(scene)

        # Save as npz
        npz_path = fixtures_dir / f"synth_scene_{i:02d}.npz"
        np.savez_compressed(
            npz_path,
            agent_hist=scene["agent_hist"],
            agent_hist_mask=scene["agent_hist_mask"],
            agent_fut=scene["agent_fut"],
            agent_fut_mask=scene["agent_fut_mask"],
            agent_type=scene["agent_type"],
            map_polylines=scene["map_polylines"],
            map_mask=scene["map_mask"],
            ego_index=np.array(scene["ego_index"]),
            scene_id=np.array(scene["scene_id"]),
        )
        print(f"✓ {npz_path}")

    # Create a minimal WOMD-shaped fixture
    womd_fixture = {
        "agent_hist": np.random.randn(8, 11, 5).astype(np.float32),
        "agent_hist_mask": np.ones((8, 11), dtype=np.bool_),
        "agent_fut": np.random.randn(8, 80, 2).astype(np.float32),
        "agent_fut_mask": np.ones((8, 80), dtype=np.bool_),
        "agent_type": np.zeros(8, dtype=np.int64),
        "map_polylines": np.random.randn(16, 20, 4).astype(np.float32),
        "map_mask": np.ones((16, 20), dtype=np.bool_),
        "ego_index": np.int64(0),
        "scene_id": np.array("womd_fixture_000000"),
    }
    validate_sample(womd_fixture)
    np.savez_compressed(fixtures_dir / "womd_fixture.npz", **womd_fixture)
    print(f"✓ {fixtures_dir / 'womd_fixture.npz'}")

    # Create a minimal AV2-shaped fixture
    av2_fixture = {
        "agent_hist": np.random.randn(6, 11, 5).astype(np.float32),
        "agent_hist_mask": np.ones((6, 11), dtype=np.bool_),
        "agent_fut": np.random.randn(6, 80, 2).astype(np.float32),
        "agent_fut_mask": np.ones((6, 80), dtype=np.bool_),
        "agent_type": np.ones(6, dtype=np.int64),
        "map_polylines": np.random.randn(12, 25, 4).astype(np.float32),
        "map_mask": np.ones((12, 25), dtype=np.bool_),
        "ego_index": np.int64(2),
        "scene_id": np.array("av2_fixture_000000"),
    }
    validate_sample(av2_fixture)
    np.savez_compressed(fixtures_dir / "av2_fixture.npz", **av2_fixture)
    print(f"✓ {fixtures_dir / 'av2_fixture.npz'}")

    print(f"\n✓ All fixtures generated in {fixtures_dir}")


if __name__ == "__main__":
    main()
