import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class CoordinateTransform(BaseTransform):
    """Express agent and map states in the reference agent's local frame."""

    def __init__(self, shuffle: bool = False) -> None:
        super().__init__()
        self.shuffle = shuffle

    def forward(self, data: HeteroData) -> HeteroData:
        agents = data["agent"]
        historical_position = agents["inp_pos"]
        historical_velocity = agents["inp_vel"]
        historical_yaw = agents["inp_yaw"]
        historical_mask = agents["input_mask"].unsqueeze(-1)
        future_position = agents["trg_pos"]
        future_velocity = agents["trg_vel"]
        future_yaw = agents["trg_yaw"]
        future_mask = agents["valid_mask"].unsqueeze(-1)

        if self.shuffle:
            valid_indices = torch.where(historical_yaw[:, -1, 0])[0]
            if len(valid_indices) > 0:
                random_index = int(torch.randint(0, len(valid_indices), (1,)).item())
                reference_index = int(valid_indices[random_index].item())
            else:
                reference_index = int(
                    torch.randint(0, historical_position.size(0), (1,)).item()
                )
        else:
            reference_index = int(agents["ta_index"])

        reference_position = historical_position[reference_index]
        reference_yaw = historical_yaw[reference_index]
        origin = reference_position[-1].unsqueeze(0)
        orientation = reference_yaw[-1]
        rotation = torch.tensor(
            [
                [torch.cos(orientation), -torch.sin(orientation)],
                [torch.sin(orientation), torch.cos(orientation)],
            ]
        )

        agents["inp_pos"] = (historical_position - origin) @ rotation * historical_mask
        agents["inp_vel"] = historical_velocity @ rotation * historical_mask
        agents["inp_yaw"] = torch.atan2(
            torch.sin(historical_yaw - orientation),
            torch.cos(historical_yaw - orientation),
        )
        agents["trg_pos"] = (future_position - origin) @ rotation * future_mask
        agents["trg_vel"] = future_velocity @ rotation * future_mask
        agents["trg_yaw"] = torch.atan2(
            torch.sin(future_yaw - orientation),
            torch.cos(future_yaw - orientation),
        )

        if "inp_acc" in agents:
            agents["inp_acc"] = agents["inp_acc"] @ rotation * historical_mask

        if "map_point" in data.node_types and "position" in data["map_point"]:
            data["map_point"]["position"] = (
                data["map_point"]["position"] - origin
            ) @ rotation

        return data
