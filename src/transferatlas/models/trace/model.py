from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import nn
from torch_geometric.data import HeteroData

from transferatlas.config import DecoderConfig, EncoderConfig, ModelConfig
from transferatlas.models.trace.layers.agent_gate import AgentGate
from transferatlas.models.trace.layers.decoders import (
    PredictionDecoder,
    ReconstructionDecoder,
)
from transferatlas.models.trace.layers.encoder import Encoder
from transferatlas.models.trace.layers.map_encoder import (
    MapEncoder,
    MultiGraphConvBipartite,
)


class _CheckpointState(TypedDict):
    state_dict: dict[str, torch.Tensor]


class _ModelCheckpoint(_CheckpointState, total=False):
    model_config: ModelConfig


class TraceModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        encoder_config = EncoderConfig(
            **config["encoder"],
            num_hidden=config["num_hidden"],
            num_latents=config["num_latents"],
            num_inputs=config["num_inputs"],
        )
        predictor_config = DecoderConfig(
            **config["predictor"],
            num_latents=config["num_latents"],
            num_outputs=config["num_outputs"],
            dt=config["dt"],
        )
        reconstructor_config = DecoderConfig(
            **config["reconstructor"],
            num_latents=config["num_latents"],
            num_outputs=config["num_outputs"],
            dt=config["dt"],
        )

        self.dt = config["dt"]

        self.encoder = Encoder(encoder_config)
        self.agent_gate = AgentGate(encoder_config)
        self.map_encoder = MapEncoder(config["lane_graph"])
        self.map2agent = MultiGraphConvBipartite(
            src_in=config["lane_graph"]["num_hidden"],
            dst_in=encoder_config["num_hidden"],
            hidden=encoder_config["num_hidden"],
            out=encoder_config["num_hidden"],
            num_layers=config["lane_graph"]["num_layers"],
            dropout=config["lane_graph"]["dropout"],
        )

        self.map_norm = nn.LayerNorm(encoder_config["num_hidden"])

        self.decoder = PredictionDecoder(predictor_config)
        self.recon = ReconstructionDecoder(reconstructor_config)

    def save_portable_checkpoint(
        self,
        model_config: ModelConfig,
        output_path: str | Path,
    ) -> Path:
        """Save the model configuration and weights used for inference."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_config": model_config,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.state_dict().items()
                },
            },
            output_path,
        )
        return output_path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: ModelConfig | None = None,
    ) -> TraceModel:
        """Construct TRACE from a portable or Lightning training checkpoint."""
        checkpoint = cast(
            _ModelCheckpoint,
            torch.load(
                Path(checkpoint_path).expanduser(),
                map_location="cpu",
                weights_only=True,
            ),
        )
        state_dict = checkpoint["state_dict"]
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        elif model_config is not None:
            state_dict = {
                name.removeprefix("model."): tensor
                for name, tensor in state_dict.items()
                if name.startswith("model.")
            }
        else:
            raise ValueError(
                "model_config is required when loading a Lightning checkpoint."
            )

        model = cls(model_config)
        _ = model.load_state_dict(state_dict, strict=True)
        return model

    def encode_env_interactions(
        self,
        x: torch.Tensor,
        data: HeteroData,
    ) -> torch.Tensor:
        edge_index = data["map_point", "to", "agent"]["edge_index"]
        edge_attr = data["map_point", "to", "agent"]["edge_attr"]

        map_embedding = self.map_encoder(data)
        wx = self.map2agent((map_embedding, x[:, -1]), edge_index, edge_attr)
        return self.map_norm(wx)

    def construct_data(
        self, data: HeteroData
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        inp_pos = data["agent"]["inp_pos"]

        x = torch.cat(
            [
                inp_pos,
                data["agent"]["inp_vel"],
                data["agent"]["inp_yaw"],
            ],
            dim=-1,
        )

        agent_types = data["agent"]["type"]

        x_emb = self.agent_gate(x, agent_types)

        x_r = inp_pos.clone()

        edge_index = data["agent"]["edge_index"]

        edge_attr = data["agent"]["edge_attr"]

        return x, x_emb, x_r, edge_index, edge_attr

    def forward(
        self,
        data: HeteroData,
        tf_prob: float = 0.0,
        encoder_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        x, x_emb, x_r, edge_index, edge_attr = self.construct_data(data)
        input_mask = data["agent"]["input_mask"]
        valid_mask = data["agent"]["valid_mask"]

        map_enc = self.encode_env_interactions(x_emb, data)
        out = self.encoder(
            x_emb, edge_index, input_mask, edge_attr, map_encoding=map_enc
        )
        if encoder_only:
            return out

        inp_vel = data["agent"]["inp_vel"]
        first_true_indices = torch.argmax(input_mask.int(), dim=1)
        first_valid_pos = x_r[torch.arange(x_r.shape[0]), first_true_indices]
        first_valid_vel = inp_vel[torch.arange(inp_vel.shape[0]), first_true_indices]
        dt = data["agent"].dt if hasattr(data["agent"], "dt") else self.dt
        x_init_guess = first_valid_pos - first_valid_vel * dt

        y = data["agent"]["trg_pos"]
        trg_edge_index = data["agent"]["trg_edge_index"]

        y_hat = self.decoder(
            x[:, -1, :2],
            trg_edge_index,
            valid_mask,
            out["z"],
            x_teacher=y,
            tf_prob=tf_prob,
            dt=dt,
        )

        x_hat = self.recon(
            x_r,
            edge_index,
            input_mask,
            out["z"],
            tf_prob=tf_prob,
            x_init=x_init_guess,
        )

        out["y"] = y
        out["y_hat"] = y_hat

        out["x"] = x[..., :2]
        out["x_hat"] = x_hat
        out["x_init"] = x_init_guess

        return out
