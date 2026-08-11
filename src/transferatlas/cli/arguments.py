from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from typing import Protocol, cast


class CommonArgs(Protocol):
    seed: int
    use_cuda: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    dry_run: bool
    small_ds: bool
    config: str
    dataset: str
    root: str


class TrainArgs(CommonArgs, Protocol):
    use_logger: bool
    store_model: bool
    add_name: str
    checkpoint: str


class LatentArgs(CommonArgs, Protocol):
    add_name: str
    checkpoint: str


def str_to_bool(value: bool | str) -> bool:
    """Parse a command-line boolean value."""
    true_values = ("yes", "true", "t", "y", "1")
    false_values = ("no", "false", "f", "n", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_values:
        return True
    if value.lower() in false_values:
        return False
    raise ArgumentTypeError("Boolean value expected.")


def _add_boolean_argument(
    parser: ArgumentParser,
    name: str,
    *aliases: str,
    default: bool,
    help: str,
) -> None:
    _ = parser.add_argument(
        name,
        *aliases,
        type=str_to_bool,
        default=default,
        const=True,
        nargs="?",
        help=help,
    )


def create_common_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    _ = parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    _add_boolean_argument(
        parser,
        "--use-cuda",
        "-uc",
        default=False,
        help="use CUDA when available (default: False)",
    )
    _ = parser.add_argument(
        "--num-workers",
        "-nw",
        type=int,
        default=1,
        help="number of DataLoader workers (default: 1)",
    )
    _add_boolean_argument(
        parser,
        "--pin-memory",
        "-pm",
        default=True,
        help="enable pinned DataLoader memory (default: True)",
    )
    _add_boolean_argument(
        parser,
        "--persistent-workers",
        "-pw",
        default=True,
        help="keep DataLoader workers alive between epochs (default: True)",
    )
    _add_boolean_argument(
        parser,
        "--dry-run",
        "-dr",
        default=True,
        help="run a small validation pass without saving artifacts (default: True)",
    )
    _add_boolean_argument(
        parser,
        "--small-ds",
        default=False,
        help="use a small dataset subset (default: False)",
    )
    _ = parser.add_argument(
        "--config",
        "-c",
        default="trace",
        help="configuration name or YAML path (default: trace)",
    )
    _ = parser.add_argument(
        "--dataset",
        "-ds",
        default="",
        help="override the configured dataset name",
    )
    _ = parser.add_argument(
        "--root",
        "-r",
        default="",
        help="override the configured dataset root",
    )
    return parser


def create_train_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train a TransferAtlas trajectory model.",
        parents=[create_common_parser()],
    )
    _add_boolean_argument(
        parser,
        "--use-logger",
        "-l",
        default=False,
        help="enable Weights & Biases logging (default: False)",
    )
    _add_boolean_argument(
        parser,
        "--store-model",
        "-s",
        default=False,
        help="save training checkpoints (default: False)",
    )
    _ = parser.add_argument(
        "--add-name",
        "-an",
        default="",
        help="suffix appended to generated checkpoint names",
    )
    _ = parser.add_argument(
        "--checkpoint",
        "--ckpt",
        default="",
        help="Lightning checkpoint from which to resume training",
    )
    return parser


def create_latent_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Compute latent Gaussian statistics for a dataset.",
        parents=[create_common_parser()],
    )
    _ = parser.add_argument(
        "--add-name",
        "-an",
        default="",
        help="suffix appended to generated artifact names",
    )
    _ = parser.add_argument(
        "--checkpoint",
        "--ckpt",
        default="",
        help="portable or Lightning model checkpoint to load",
    )
    return parser


def parse_train_args(argv: Sequence[str] | None = None) -> TrainArgs:
    parsed = create_train_parser().parse_args(argv)
    return cast(TrainArgs, cast(object, parsed))


def parse_latent_args(argv: Sequence[str] | None = None) -> LatentArgs:
    parsed = create_latent_parser().parse_args(argv)
    return cast(LatentArgs, cast(object, parsed))
