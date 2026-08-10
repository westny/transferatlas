import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv


class MultiGraphConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [in_channels] + [hidden_channels] * (num_layers - 1)
        next_dimensions = [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            GraphConv(source, target, aggr="add", bias=True)
            for source, target in zip(dimensions, next_dimensions)
        )
        self.acts = nn.ModuleList(nn.GELU() for _ in range(num_layers))
        self.drops = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_weight = None if edge_attr is None else edge_attr[:, 0]
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if index < len(self.convs) - 1:
                x = self.drops[index](self.acts[index](x))
        return x


class MultiGraphConvBipartite(nn.Module):
    def __init__(
        self,
        src_in: int,
        dst_in: int,
        hidden: int,
        out: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            GraphConv(
                (src_in, dst_in if index == 0 else hidden),
                out if index == num_layers - 1 else hidden,
                aggr="add",
                bias=True,
            )
            for index in range(num_layers)
        )
        self.acts = nn.ModuleList(
            nn.Identity() if index == num_layers - 1 else nn.GELU()
            for index in range(num_layers)
        )
        self.drops = nn.ModuleList(
            nn.Dropout(dropout) if index < num_layers - 1 else nn.Identity()
            for index in range(num_layers)
        )

    def forward(
        self,
        x_pair: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source, target = x_pair
        edge_weight = None if edge_attr is None else edge_attr[:, 0]
        for conv, activation, dropout in zip(self.convs, self.acts, self.drops):
            target = conv((source, target), edge_index, edge_weight=edge_weight)
            target = dropout(activation(target))
        return target


class MapEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        edge_features = config["edge_feats"]
        dropout = config["dropout"]
        self.num_edge_feats = edge_features
        self.scale = nn.Sequential(nn.Linear(edge_features, 1))
        self.dropout = nn.Dropout(dropout)
        self.net = MultiGraphConv(
            in_channels=config["num_inputs"],
            hidden_channels=config["num_hidden"],
            out_channels=config["num_hidden"],
            num_layers=config["num_layers"],
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_type = data["map_point", "to", "map_point"]["type"].squeeze().long()
        edge_attr = nn.functional.one_hot(
            edge_type, num_classes=self.num_edge_feats
        ).float()
        edge_attr = self.scale(edge_attr)
        encoded = self.net(
            data["map_point"]["position"],
            data["map_point", "to", "map_point"]["edge_index"],
            edge_attr,
        )
        return self.dropout(encoded)
