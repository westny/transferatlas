from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData


def collate_trajectory_batch(data_list: list[HeteroData]) -> Batch:
    """Pad variable-length trajectories and combine them into a graph batch."""
    if not data_list:
        raise ValueError("Cannot collate an empty trajectory batch.")

    padded_data = []

    max_inp_len = max(data["agent"]["inp_pos"].size(1) for data in data_list)
    max_trg_len = max(data["agent"]["trg_pos"].size(1) for data in data_list)

    for data in data_list:
        data = copy.deepcopy(data)

        data[("map_point", "to", "map_point")].type = (
            data[("map_point", "to", "map_point")].type.to(torch.long).view(-1, 1)
        )

        if "rec_id" in data:
            del data["rec_id"]

        agent_data = data["agent"]

        for field in ["ids", "intention", "sa_mask", "ma_mask", "inp_r1", "inp_r2"]:
            if field in agent_data:
                del agent_data[field]

        # Pad input/target sequences
        for key in list(agent_data.keys()):
            value = agent_data[key]
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                current_length = value.size(1)

                if key.startswith(("inp_", "input_")) and current_length < max_inp_len:
                    pad = [0] * (2 * (value.dim() - 2)) + [
                        max_inp_len - current_length,
                        0,
                    ]
                    agent_data[key] = F.pad(value, pad)

                elif (
                    key.startswith(("trg_", "valid_")) and current_length < max_trg_len
                ):
                    pad = [0] * (2 * (value.dim() - 2)) + [
                        0,
                        max_trg_len - current_length,
                    ]
                    agent_data[key] = F.pad(value, pad)

        padded_data.append(data)

    return Batch.from_data_list(padded_data)
