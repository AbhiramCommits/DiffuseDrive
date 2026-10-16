"""Data loading and dataset building."""

from torch.utils.data import Dataset

from .av2 import Argoverse2Dataset
from .collate import collate_batch
from .synth import SynthDriveDataset
from .womd import WOMDScenarioDataset


def build_dataset(cfg) -> Dataset:
    """Build dataset based on config source."""
    source = cfg.source.lower()

    if source == "synth":
        return SynthDriveDataset(
            num_scenes=cfg.num_train_scenes,
            T_hist=cfg.T_hist,
            T_fut=cfg.T_fut,
            seed=cfg.synthetic_seed,
        )
    elif source == "womd":
        return WOMDScenarioDataset(
            data_root=cfg.data_root,
            num_scenes=cfg.num_train_scenes,
            T_hist=cfg.T_hist,
            T_fut=cfg.T_fut,
            split="train",
        )
    elif source == "av2":
        return Argoverse2Dataset(
            data_root=cfg.data_root,
            num_scenes=cfg.num_train_scenes,
            T_hist=cfg.T_hist,
            T_fut=cfg.T_fut,
            split="train",
        )
    else:
        raise ValueError(f"Unknown data source: {source}")


__all__ = [
    "SynthDriveDataset",
    "WOMDScenarioDataset",
    "Argoverse2Dataset",
    "build_dataset",
    "collate_batch",
]
