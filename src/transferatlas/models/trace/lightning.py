from __future__ import annotations

from logging import getLogger

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius, radius_graph
from torch_geometric.utils import dropout_edge, scatter, subgraph, to_undirected

from transferatlas.metrics import MinADE, MinFDE

logger = getLogger(__name__)


def compute_batched_scene_statistics(
    z: torch.Tensor,
    batch: torch.Tensor,
    eps: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute per-scene means, sample covariances, and agent counts."""
    if z.ndim != 2:
        raise ValueError(f"Expected z to have shape [N, D], got {tuple(z.shape)}.")
    if batch.ndim != 1 or batch.numel() != z.size(0):
        raise ValueError(
            "Expected batch to have shape [N] matching z, got "
            f"{tuple(batch.shape)} for {z.size(0)} rows."
        )
    if batch.numel() == 0:
        raise ValueError("At least one agent embedding is required.")

    num_scenes = int(batch.max().item()) + 1
    num_agents, dim = z.shape
    ones = torch.ones(num_agents, device=z.device, dtype=z.dtype)
    scene_count = scatter(
        ones,
        batch,
        dim=0,
        dim_size=num_scenes,
        reduce="sum",
    )
    scene_mu = scatter(
        z,
        batch,
        dim=0,
        dim_size=num_scenes,
        reduce="mean",
    )

    centered = z - scene_mu[batch]
    outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)
    cov_num = scatter(
        outer,
        batch,
        dim=0,
        dim_size=num_scenes,
        reduce="sum",
    )
    denom = (scene_count - 1.0).clamp(min=1.0).view(-1, 1, 1)
    scene_cov = cov_num / denom
    scene_cov = 0.5 * (scene_cov + scene_cov.transpose(-1, -2))

    if eps > 0.0:
        eye = torch.eye(dim, device=z.device, dtype=z.dtype).unsqueeze(0)
        scene_cov = scene_cov + eps * eye

    return {
        "mean": scene_mu,
        "covariance": scene_cov,
        "count": scene_count,
    }


class TraceLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, config: dict) -> None:
        super().__init__()
        self.model = model
        self.max_epochs = config["epochs"]
        self.learning_rate = config["lr"]
        self.initial_teacher_forcing_probability = config.get("tf_init_p", 1.0)
        self.agent_radius = config.get("radius", 50.0)
        self.map_radius = config.get("map_radius", 50.0)

        self.agent_neighbors = config.get("agent_neighbors", 10)
        self.map_neighbors = config.get("map_neighbors", 100)
        self.edge_dropout = config.get("edge_dropout", 0.5)

        self.teacher_forcing_epochs = config.get(
            "teacher_force_epochs",
            self.max_epochs // 4,
        )
        self.alpha = config.get("alpha", 1.0)

        self.save_hyperparameters(ignore=["model"])

        self.min_ade = MinADE()
        self.min_fde = MinFDE()

    def forward(self, data: HeteroData, tf_prob: float = 0.0) -> dict:
        data = self.prepare_graph(data)
        out = self.model(data, tf_prob=tf_prob)
        return out

    def create_agent_edges(self, pos, batch, mask):
        pos = pos[:, -1]

        edge_index = radius_graph(
            x=pos,
            r=self.agent_radius,
            batch=batch,
            loop=True,
            max_num_neighbors=self.agent_neighbors,
        )
        edge_index = subgraph(subset=mask[:, -1], edge_index=edge_index)[0]
        edge_index = to_undirected(edge_index)

        if self.training:
            # Separate self-edges
            self_edges = edge_index[:, edge_index[0] == edge_index[1]]
            non_self_edges = edge_index[:, edge_index[0] != edge_index[1]]

            # Apply dropout to non-self edges
            non_self_edges, _ = dropout_edge(non_self_edges, p=self.edge_dropout)

            # Combine self-edges with the remaining non-self edges
            edge_index = torch.cat([self_edges, non_self_edges], dim=1)

        dist = torch.linalg.norm(
            pos[edge_index[0]] - pos[edge_index[1]], dim=-1, keepdim=True
        )

        # RBF kernel
        sigma = self.agent_radius / 2.0
        edge_attrs = torch.exp(-((dist / sigma) ** 2))

        return edge_index, edge_attrs

    def prepare_graph(self, data: HeteroData) -> HeteroData:
        batch = data["agent"]["batch"]
        inp_pos = data["agent"]["inp_pos"]
        inp_mask = data["agent"]["input_mask"]
        trg_pos = data["agent"]["trg_pos"]

        edge_indices, edge_attributes = self.create_agent_edges(
            inp_pos, batch, inp_mask
        )

        seq_len = trg_pos.shape[1]
        edge_index = edge_indices.clone()
        trg_edge_indices = [edge_index for _ in range(seq_len)]

        data["agent"]["edge_index"] = edge_indices
        data["agent"]["edge_attr"] = edge_attributes
        data["agent"]["trg_edge_index"] = trg_edge_indices

        map_batch = data["map_point"]["batch"]
        map_pos = data["map_point"]["position"]

        edge_index_m2a = radius(
            x=inp_pos[:, -1],
            y=map_pos,
            r=self.map_radius,
            batch_x=batch,
            batch_y=map_batch,
            max_num_neighbors=self.map_neighbors,
        )

        if self.training:
            edge_index_m2a, _ = dropout_edge(edge_index_m2a, p=self.edge_dropout)

        dist = torch.linalg.norm(
            inp_pos[edge_index_m2a[1], -1] - map_pos[edge_index_m2a[0]],
            dim=-1,
            keepdim=True,
        )

        sigma = self.map_radius / 2.0
        edge_attributes_m2a = torch.exp(-((dist / sigma) ** 2))

        data["map_point", "to", "agent"]["edge_index"] = edge_index_m2a
        data["map_point", "to", "agent"]["edge_attr"] = edge_attributes_m2a

        return data

    @staticmethod
    def reconstruction_loss(data: HeteroData, out: dict) -> torch.Tensor:
        mask = data["agent"]["input_mask"]

        trg = out["x"]
        pred = out["x_hat"]

        num_valid_steps = mask.sum(-1)
        norm = torch.linalg.norm(pred - trg, dim=-1)
        masked_norm = norm * mask

        scored_agents = num_valid_steps > 0
        summed_loss = (
            masked_norm[scored_agents].sum(-1) / num_valid_steps[scored_agents]
        )

        loss = summed_loss.mean()
        return loss

    @staticmethod
    def prediction_loss(data: HeteroData, out: dict) -> torch.Tensor:
        mask = data["agent"]["valid_mask"]

        trg = out["y"]
        pred = out["y_hat"]

        num_valid_steps = mask.sum(-1)
        norm = torch.linalg.norm(pred - trg, dim=-1)
        masked_norm = norm * mask

        scored_agents = num_valid_steps > 0
        summed_loss = (
            masked_norm[scored_agents].sum(-1) / num_valid_steps[scored_agents]
        )

        loss = summed_loss.mean()
        return loss

    def training_step(self, data: HeteroData) -> torch.Tensor:
        # valid_mask = data['agent']['valid_mask']
        if self.teacher_forcing_epochs <= 0:
            tf_prob = 0.0
        else:
            remaining_fraction = (
                self.teacher_forcing_epochs - self.current_epoch
            ) / self.teacher_forcing_epochs
            tf_prob = max(
                0.0,
                self.initial_teacher_forcing_probability * remaining_fraction,
            )
        out = self(data, tf_prob=tf_prob)
        z = out["z"]

        rec_loss = self.reconstruction_loss(data, out)
        prediction_loss = self.prediction_loss(data, out)
        loss = rec_loss + self.alpha * prediction_loss

        metrics_dict = {
            "train_loss": loss,
            "train_rec_loss": rec_loss,
            "train_pred_loss": prediction_loss,
        }

        self.log_dict(
            metrics_dict,
            on_step=False,
            on_epoch=True,
            batch_size=z.size(0),
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, data: HeteroData, *args) -> None:
        mask = data["agent"]["valid_mask"]
        out = self(data)
        z = out["z"]
        rec_loss = self.reconstruction_loss(data, out)
        prediction_loss = self.prediction_loss(data, out)
        val_loss = rec_loss + self.alpha * prediction_loss

        trg = out["y"]
        pred = out["y_hat"]

        self.min_ade.update(pred, trg, mask=mask)
        self.min_fde.update(pred, trg, mask=mask)

        metric_dict = {
            "val_loss": val_loss,
            "val_rec_loss": rec_loss,
            "val_pred_loss": prediction_loss,
            "val_ade": self.min_ade,
            "val_fde": self.min_fde,
        }

        self.log_dict(
            metric_dict,
            on_step=False,
            on_epoch=True,
            batch_size=z.size(0),
            prog_bar=False,
            sync_dist=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        batches = len(self.trainer.datamodule.train_dataloader())  # type: ignore
        total_steps = self.max_epochs * batches

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": cosine,
                "interval": "step",
                "frequency": 1,
            },
        }

    def compute_scene_statistics(
        self, data: HeteroData, eps: float = 0.0
    ) -> dict[str, torch.Tensor]:
        """Compute full scene-level Gaussian statistics on the CPU."""
        was_training = self.training
        self.eval()

        with torch.no_grad():
            data = self.prepare_graph(data)
            out = self.model(data, encoder_only=True)

            z = out["z"]  # [N, D]
            batch = data["agent"]["batch"]  # [N]

            statistics = compute_batched_scene_statistics(z, batch, eps=eps)
            result = {key: value.detach().cpu() for key, value in statistics.items()}

        if was_training:
            self.train()

        return result
