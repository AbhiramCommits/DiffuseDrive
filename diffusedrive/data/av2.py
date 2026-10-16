"""Argoverse 2 motion forecasting loader with graceful degradation."""

from typing import Any, Dict

from torch.utils.data import Dataset

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class Argoverse2Dataset(Dataset):
    """Load AV2 parquet scenarios to canonical schema."""

    def __init__(
        self,
        data_root: str,
        num_scenes: int = 100,
        T_hist: int = 11,
        T_fut: int = 80,
        split: str = "train",
    ):
        if not PYARROW_AVAILABLE:
            raise ImportError(
                "pandas/pyarrow not installed. Install with:\n"
                "  pip install pandas pyarrow\n"
                "Then point --data-root to AV2 parquet directory."
            )

        self.data_root = data_root
        self.num_scenes = num_scenes
        self.T_hist = T_hist
        self.T_fut = T_fut
        self.split = split

        # Stub: in real use, would enumerate parquet files
        self.file_paths = []

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and convert AV2 parquet to schema."""
        # Stub implementation
        raise NotImplementedError(
            "AV2 loader stub. Implement parquet parsing in __getitem__."
        )
