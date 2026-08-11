from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from typing import Literal, Protocol, cast

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from transferatlas.data.collate import collate_trajectory_batch
from transferatlas.data.dataset import BalancedConcatDataset, TrajectoryDataset
from transferatlas.data.transforms import CoordinateTransform


class _WeightedDataset(Protocol):
    sample_weights: Sequence[float]


class TrajectoryDataModule(LightningDataModule):
    train_dataset: Dataset | None = None
    validation_dataset: Dataset | None = None
    test_dataset: Dataset | None = None

    def __init__(self, config: dict, args: Namespace) -> None:
        super().__init__()
        self.root = config["root"]
        dataset_names = config["name"]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if not isinstance(dataset_names, list) or not all(
            isinstance(name, str) for name in dataset_names
        ):
            raise TypeError("Dataset names must be a string or a list of strings.")
        if not dataset_names:
            raise ValueError("At least one dataset name is required.")

        self.dataset_names = dataset_names
        self.batch_size = config["batch_size"]
        self.randomize_reference_agent = config["shuffle"]
        self.small_data = args.small_ds
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = args.persistent_workers
        self.train_transform = CoordinateTransform(
            shuffle=self.randomize_reference_agent
        )
        self.validation_transform = CoordinateTransform()

    @property
    def _keep_workers_alive(self) -> bool:
        return self.persistent_workers and self.num_workers > 0

    @staticmethod
    def _require_dataset(dataset: Dataset | None, split: str) -> Dataset:
        if dataset is None:
            raise RuntimeError(
                f"The {split} dataset is unavailable; call setup() for that stage first."
            )
        return dataset

    def create_dataset(
        self,
        dataset_name: str,
        split: Literal["train", "val", "test"],
        transform: CoordinateTransform,
    ) -> TrajectoryDataset:
        return TrajectoryDataset(
            root=self.root,
            dataset=dataset_name,
            split=split,
            transform=transform,
            small_data=self.small_data,
        )

    def _combined_dataset(
        self,
        split: Literal["train", "val", "test"],
        transform: CoordinateTransform,
    ) -> BalancedConcatDataset:
        return BalancedConcatDataset(
            [self.create_dataset(name, split, transform) for name in self.dataset_names]
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = self._combined_dataset("train", self.train_transform)
            self.validation_dataset = self._combined_dataset(
                "val", self.validation_transform
            )
        elif stage == "validate":
            self.validation_dataset = self._combined_dataset(
                "val", self.validation_transform
            )

        if stage in {None, "test"}:
            self.test_dataset = self._combined_dataset(
                "test", self.validation_transform
            )

    def train_dataloader(self) -> DataLoader:
        dataset = self._require_dataset(self.train_dataset, "train")
        weights = cast(_WeightedDataset, dataset).sample_weights
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_trajectory_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._keep_workers_alive,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.validation_dataset, "validation"),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_trajectory_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._keep_workers_alive,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.test_dataset, "test"),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_trajectory_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._keep_workers_alive,
        )
