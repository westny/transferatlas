from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import subgraph

from transferatlas.config import DecoderConfig
from transferatlas.models.trace.layers.min_gru_gnn_cell import GRUGNNCell


class PredictionDecoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        num_hidden = config["num_latents"]
        num_outputs = config["num_outputs"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]
        self.dt = config["dt"]

        self.output = nn.Sequential(
            nn.Linear(num_hidden, num_hidden // 2),
            nn.SiLU(),
            nn.Linear(num_hidden // 2, num_outputs),
        )
        self.decoder = GRUGNNCell(num_outputs, num_hidden, num_layers, dropout)

    def forward(
        self,
        x_init: torch.Tensor,
        trg_edge_index: list[torch.Tensor],
        valid_mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
        x_teacher: torch.Tensor | None = None,
        tf_prob: float = 0.0,
        dt: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if dt is None:
            dt = self.dt

        output = []
        x = x_init
        for i, edge_index in enumerate(trg_edge_index):
            hidden = self.decoder(x, edge_index, hidden)
            velocity = self.output(hidden)
            x_next = x + velocity * dt
            output.append(x_next)

            use_teacher_forcing = tf_prob > 0.0 and torch.rand(1).item() < tf_prob
            if use_teacher_forcing and x_teacher is not None:
                mask = valid_mask[:, i].view(-1, 1)
                x = mask * x_teacher[:, i] + (~mask) * x_next
            else:
                x = x_next

        return torch.stack(output, dim=1)


class ReconstructionDecoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        num_hidden = config["num_latents"]
        num_outputs = config["num_outputs"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]
        self.dt = config["dt"]

        self.output = nn.Sequential(
            nn.Linear(num_hidden, num_hidden // 2),
            nn.SiLU(),
            nn.Linear(num_hidden // 2, num_outputs),
        )
        self.decoder = GRUGNNCell(num_outputs, num_hidden, num_layers, dropout)
        self.init_token = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(
        self,
        x_teacher: torch.Tensor,
        edge_index: torch.Tensor,
        input_mask: torch.Tensor,
        hidden: torch.Tensor,
        tf_prob: float = 0.0,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = []
        x = (
            x_init
            if x_init is not None
            else self.init_token.expand(x_teacher.size(0), -1).clone()
        )

        for i in range(x_teacher.size(1)):
            mask = input_mask[:, i]
            x_i = x[mask]
            hidden_now = hidden[mask]
            edge_index_i = subgraph(
                subset=mask, edge_index=edge_index, relabel_nodes=True
            )[0]
            hidden_next = self.decoder(x_i, edge_index_i, hidden_now)

            velocity = self.output(hidden_next)
            x_i = x_i + velocity * self.dt
            x_next = x.clone()
            x_next[mask] = x_i
            output.append(x_next)

            new_hidden = hidden.clone()
            new_hidden[mask] = hidden_next
            hidden = new_hidden

            use_teacher_forcing = tf_prob > 0.0 and torch.rand(1).item() < tf_prob
            if use_teacher_forcing:
                expanded_mask = mask.view(-1, 1)
                x = expanded_mask * x_teacher[:, i] + (~expanded_mask) * x_next
            else:
                x = x_next

        return torch.stack(output, dim=1)
