# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import os
import pickle
import warnings
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.data import Dataset, HeteroData

# Ignore FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class DatasetLabel(IntEnum):
    ROUND = 0
    IND = 1
    UNID = 2
    SIND = 3
    US101 = 4
    I80 = 5
    HIGHD = 6
    EXID = 7
    A43 = 8
    AD4CHE = 9
    INTERACT = 10
    APOLLO = 11
    ARGOVERSE = 12
    AV2 = 13
    NUSCENES = 14
    WAYMO = 15
    LYFT = 16
    OPENDD = 17
    ETH = 18
    HOTEL = 19
    UNIV = 20
    ZARA1 = 21
    ZARA2 = 22
    VOD = 23

    @classmethod
    def from_name(cls, name: str) -> DatasetLabel:
        normalized_name = name.strip().lower()
        lookup = {
            "round": cls.ROUND,
            "ind": cls.IND,
            "unid": cls.UNID,
            "sind": cls.SIND,
            "us101": cls.US101,
            "i80": cls.I80,
            "highd": cls.HIGHD,
            "exid": cls.EXID,
            "a43": cls.A43,
            "ad4che": cls.AD4CHE,
            "interact": cls.INTERACT,
            "apollo": cls.APOLLO,
            "argoverse": cls.ARGOVERSE,
            "av2": cls.AV2,
            "argoverse2": cls.AV2,
            "nuscenes": cls.NUSCENES,
            "waymo": cls.WAYMO,
            "lyft": cls.LYFT,
            "opendd": cls.OPENDD,
            "eth": cls.ETH,
            "hotel": cls.HOTEL,
            "univ": cls.UNIV,
            "zara1": cls.ZARA1,
            "zara2": cls.ZARA2,
            "vod": cls.VOD,
        }
        if normalized_name not in lookup:
            raise ValueError(f"Unknown dataset name: {name}")
        return lookup[normalized_name]


def dataset_group_name(dataset_names: str | Sequence[str]) -> str:
    """Return the artifact name for one dataset or a multi-dataset group."""
    if isinstance(dataset_names, str):
        return dataset_names
    if len(dataset_names) == 1:
        return dataset_names[0]
    return "multi"


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset: str,
        split: Literal["train", "val", "test"],
        transform: Callable | None = None,
        small_data: bool = False,
    ) -> None:
        super().__init__(
            root=root, transform=transform, pre_transform=None, pre_filter=None
        )

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset split: {split}")

        self.root = root
        self.dataset = dataset
        self.split = split
        self.path = os.path.join(self.root, self.dataset, self.split)
        self.files = os.listdir(self.path)
        self.files = sorted(
            self.files
        )  # sort files for consistency across operating systems

        self.label = DatasetLabel.from_name(self.dataset)

        if small_data:
            self.files = self.files[:100]

        self._num_samples = len(self.files)

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        with open(os.path.join(self.path, self.files[idx]), "rb") as f:
            data = pickle.load(f)
            ds_label = torch.tensor([self.label.value], dtype=torch.long).expand(
                data["agent"]["num_nodes"]
            )
            data["agent"]["ds_label"] = ds_label
            return HeteroData(data)


class BalancedConcatDataset(ConcatDataset):
    """Concatenated datasets with sampling weights balanced by source size."""

    def __init__(self, datasets: list[TrajectoryDataset]) -> None:
        super().__init__(datasets)
        self._balanced_datasets = datasets
        self.alpha = 0.5
        self.sample_weights = self._compute_weights()

    def _compute_weights(self) -> list[float]:
        dataset_ids = []
        for index, dataset in enumerate(self._balanced_datasets):
            dataset_ids.extend([index] * len(dataset))

        source_counts = np.bincount(dataset_ids)
        source_weights = 1.0 / (source_counts**self.alpha)
        return [float(source_weights[index]) for index in dataset_ids]
