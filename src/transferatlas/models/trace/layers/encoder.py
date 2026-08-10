import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import subgraph

from transferatlas.models.trace.layers.min_gru_gnn_cell import GRUGNNCell


class Encoder(nn.Module):
    """Spherical TRACE scene encoder."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        num_hidden = config["num_hidden"]
        num_latents = config["num_latents"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]

        self.layer = GRUGNNCell(
            num_hidden,
            num_hidden,
            num_layers,
            dropout,
        )
        self.combine = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_latents),
        )

        std = 1 / (num_hidden**0.5)
        self.h_init = nn.Parameter(torch.zeros(1, num_hidden).uniform_(-std, std))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        input_mask: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        map_encoding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.h_init.expand(x.shape[0], -1)
        for index in range(x.size(1)):
            mask = input_mask[:, index]
            edge_index_i, edge_attr_i = subgraph(
                subset=mask,
                edge_index=edge_index,
                edge_attr=edge_attr,
                relabel_nodes=True,
            )
            hidden_next = self.layer(
                x[mask, index],
                edge_index_i,
                hidden[mask],
                edge_attr=edge_attr_i,
            )
            hidden = hidden.clone()
            hidden[mask] = hidden_next

        if map_encoding is None:
            raise ValueError("TRACE requires map encoding.")

        hidden = self.combine(torch.cat([hidden, map_encoding], dim=-1))
        latent = F.normalize(self.output(hidden), p=2, dim=-1, eps=1e-8)
        return {"z": latent, "hidden": hidden}
