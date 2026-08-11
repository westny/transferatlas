from __future__ import annotations

from typing import TypedDict


class RecurrentConfig(TypedDict):
    num_layers: int
    dropout: float


class EncoderConfig(RecurrentConfig):
    num_hidden: int
    num_latents: int
    num_inputs: int


class DecoderConfig(RecurrentConfig):
    num_latents: int
    num_outputs: int
    dt: float


class LaneGraphConfig(TypedDict):
    num_inputs: int
    num_hidden: int
    num_layers: int
    edge_feats: int
    dropout: float


class ModelConfig(TypedDict):
    name: str
    dt: float
    num_inputs: int
    num_outputs: int
    num_hidden: int
    num_latents: int
    encoder: RecurrentConfig
    predictor: RecurrentConfig
    reconstructor: RecurrentConfig
    lane_graph: LaneGraphConfig


class DataModuleConfig(TypedDict):
    root: str
    name: str | list[str]
    batch_size: int
    shuffle: bool


class RequiredTrainingConfig(TypedDict):
    epochs: int
    lr: float


class TrainingConfig(RequiredTrainingConfig, total=False):
    dataset: str
    tf_init_p: float
    radius: float
    map_radius: float
    agent_neighbors: int
    map_neighbors: int
    edge_dropout: float
    teacher_force_epochs: int
    alpha: float


class ExperimentConfig(TypedDict):
    experiment_name: str
    model: ModelConfig
    datamodule: DataModuleConfig
    training: TrainingConfig
