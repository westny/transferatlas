from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import HeteroData

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

CHECKPOINT_FORMAT_VERSION = 1


class TraceModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        encoder_config = dict(config["encoder"])
        predictor_config = dict(config["predictor"])
        reconstructor_config = dict(config["reconstructor"])
        encoder_config.update(
            num_hidden=config["num_hidden"],
            num_latents=config["num_latents"],
            num_inputs=config["num_inputs"],
        )
        decoder_shared = {
            "num_latents": config["num_latents"],
            "num_outputs": config["num_outputs"],
            "dt": config["dt"],
        }
        predictor_config.update(decoder_shared)
        reconstructor_config.update(decoder_shared)

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
        model_config: dict,
        output_path: str | Path,
    ) -> Path:
        """Save the model configuration and weights used for inference."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
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
        model_config: dict | None = None,
    ) -> TraceModel:
        """Construct TRACE from a portable or Lightning training checkpoint."""
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("format_version") == CHECKPOINT_FORMAT_VERSION:
            model_config = checkpoint["model_config"]
            state_dict = checkpoint["state_dict"]
        elif "state_dict" in checkpoint:
            if model_config is None:
                raise ValueError(
                    "model_config is required when loading a Lightning checkpoint."
                )
            state_dict = {
                name.removeprefix("model."): tensor
                for name, tensor in checkpoint["state_dict"].items()
                if name.startswith("model.")
            }
        else:
            raise ValueError(
                "Checkpoint must contain either portable model data or a Lightning "
                "state_dict."
            )

        if model_config is None:
            raise ValueError("Checkpoint does not define a model configuration.")
        model = cls(model_config)
        model.load_state_dict(state_dict, strict=True)
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

    def construct_data(self, data: HeteroData):
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
    ) -> dict:
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
