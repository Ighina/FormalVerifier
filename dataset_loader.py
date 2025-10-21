"""Dataset loader for various mathematical problem datasets."""

from typing import Iterator, Dict, Any, Optional
from datasets import load_dataset


class DatasetLoader:
    """Loader for mathematical problem datasets."""

    def __init__(self, dataset_name: str = "gsm8k", split: str = "train", config: Optional[str] = None):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: Name of the dataset (default: "gsm8k")
            split: Dataset split to load (default: "train")
            config: Dataset configuration/subset (for gsm8k, use "main")
        """
        self.dataset_name = dataset_name
        self.split = split
        self.config = config or ("main" if dataset_name == "gsm8k" else None)
        self.dataset = None

    def load(self):
        """Load the dataset from HuggingFace."""
        if self.config:
            self.dataset = load_dataset(self.dataset_name, self.config, split=self.split)
        else:
            self.dataset = load_dataset(self.dataset_name, split=self.split)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset."""
        if self.dataset is None:
            self.load()

        for idx, item in enumerate(self.dataset):
            yield self._process_item(idx, item)

    def _process_item(self, idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a dataset item into a standardized format.

        Args:
            idx: Index of the item
            item: Raw item from the dataset

        Returns:
            Standardized item with 'id', 'problem', and 'solution' keys
        """
        if self.dataset_name == "gsm8k":
            return {
                "id": idx,
                "problem": item["question"],
                "solution": item["answer"],
                "original": item
            }
        else:
            # Generic fallback - assumes 'problem' and 'solution' keys exist
            return {
                "id": idx,
                "problem": item.get("problem", item.get("question", "")),
                "solution": item.get("solution", item.get("answer", "")),
                "original": item
            }

    def get_subset(self, start: int = 0, end: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Get a subset of the dataset.

        Args:
            start: Starting index (inclusive)
            end: Ending index (exclusive), None for end of dataset

        Yields:
            Processed dataset items
        """
        if self.dataset is None:
            self.load()

        end = end or len(self.dataset)
        for idx in range(start, min(end, len(self.dataset))):
            yield self._process_item(idx, self.dataset[idx])

    def get_item(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item by index.

        Args:
            idx: Index of the item

        Returns:
            Processed dataset item
        """
        if self.dataset is None:
            self.load()

        return self._process_item(idx, self.dataset[idx])

    def get_batches(self, batch_size: int, start: int = 0, end: Optional[int] = None) -> Iterator[list[Dict[str, Any]]]:
        """
        Iterate through the dataset in batches.

        Args:
            batch_size: Number of items per batch
            start: Starting index (inclusive)
            end: Ending index (exclusive), None for end of dataset

        Yields:
            Batches of processed dataset items
        """
        if self.dataset is None:
            self.load()

        end = end or len(self.dataset)
        for batch_start in range(start, min(end, len(self.dataset)), batch_size):
            batch_end = min(batch_start + batch_size, min(end, len(self.dataset)))
            batch = []
            for idx in range(batch_start, batch_end):
                batch.append(self._process_item(idx, self.dataset[idx]))
            yield batch
