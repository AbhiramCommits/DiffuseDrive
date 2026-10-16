"""WOMD scenario loader with graceful degradation."""

from typing import Any, Dict

from torch.utils.data import Dataset

try:
    import tensorflow as tf  # noqa: F401 - availability check
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class WOMDScenarioDataset(Dataset):
    """Load WOMD tf.Example/scenario protos to canonical schema."""

    def __init__(
        self,
        data_root: str,
        num_scenes: int = 100,
        T_hist: int = 11,
        T_fut: int = 80,
        split: str = "train",
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "tensorflow is not installed. Install with:\n"
                "  pip install tensorflow\n"
                "Then point --data-root to WOMD scenario proto directory."
            )

        self.data_root = data_root
        self.num_scenes = num_scenes
        self.T_hist = T_hist
        self.T_fut = T_fut
        self.split = split

        # Stub: in real use, would enumerate tfrecord files
        self.file_paths = []

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and convert WOMD scenario proto to schema."""
        # Stub implementation
        raise NotImplementedError(
            "WOMD loader stub. Implement scenario proto parsing in __getitem__."
        )
