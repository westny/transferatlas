from __future__ import annotations

import torch
import torch_geometric.nn as pyg_nn
from torch import nn


def _graphconv_stack(
    input_size: int,
    output_size: int,
    hidden_size: int,
    layers: int,
    dropout: float,
) -> pyg_nn.Sequential:
    modules: list = []
    if layers == 1:
        modules.append(
            (
                pyg_nn.GraphConv(input_size, output_size, aggr="mean", bias=False),
                "x, edge_index, edge_attr -> x",
            )
        )
    else:
        modules.append(
            (
                pyg_nn.GraphConv(hidden_size, output_size, aggr="mean", bias=False),
                "x, edge_index, edge_attr -> x",
            )
        )
        remaining_layers = layers - 1
        while remaining_layers > 1:
            modules.insert(0, nn.SiLU(inplace=True))
            modules.insert(
                0,
                (
                    pyg_nn.GraphConv(hidden_size, hidden_size, aggr="mean", bias=True),
                    "x, edge_index, edge_attr -> x",
                ),
            )
            remaining_layers -= 1
        modules.insert(0, nn.SiLU(inplace=True))
        modules.insert(0, nn.Dropout(dropout))
        modules.insert(
            0,
            (
                pyg_nn.GraphConv(input_size, hidden_size, aggr="mean", bias=True),
                "x, edge_index, edge_attr -> x",
            ),
        )
    return pyg_nn.Sequential("x, edge_index, edge_attr", modules)


class GRUGNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        gate_size = 2 * hidden_size
        init_std = 1.0 / (hidden_size**0.5)

        self.Wx = _graphconv_stack(
            input_size,
            gate_size,
            hidden_size,
            num_layers,
            dropout,
        )
        self.bias = nn.Parameter(torch.empty(gate_size).uniform_(-init_std, init_std))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_size**0.5)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter, -std, std)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        gate, candidate = torch.split(
            self.Wx(x, edge_index, edge_attr), self.hidden_size, dim=1
        )
        gate_bias, candidate_bias = torch.split(self.bias, self.hidden_size, dim=0)
        update = torch.sigmoid(gate + gate_bias)
        proposal = torch.tanh(candidate + candidate_bias)
        return (1 - update) * hidden + update * proposal
