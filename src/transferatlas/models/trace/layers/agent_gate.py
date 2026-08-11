from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import one_hot

from transferatlas.config import EncoderConfig


class AgentGate(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_hidden = config["num_hidden"]
        self.num_types = 9

        self.hyper_gate = nn.Linear(self.num_types, num_hidden)
        self.hyper_bias = nn.Linear(self.num_types, num_hidden, bias=False)
        self.embed = nn.Linear(num_inputs, num_hidden)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, num_inputs)   # Agent features
            c: (N)                  # Agent type
        Returns:
            x: (N, num_hidden)
        """

        c_1hot = one_hot(c, num_classes=self.num_types).float().unsqueeze(1)
        gate = torch.sigmoid(self.hyper_gate(c_1hot))
        bias = self.hyper_bias(c_1hot)

        x = self.embed(x) * gate + bias

        return x
