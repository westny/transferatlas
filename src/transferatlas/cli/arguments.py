from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Sequence


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
    parser.add_argument(
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
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    _add_boolean_argument(
        parser,
        "--use-cuda",
        "-uc",
        default=False,
        help="use CUDA when available (default: False)",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--config",
        "-c",
        default="trace",
        help="configuration name or YAML path (default: trace)",
    )
    parser.add_argument(
        "--dataset",
        "-ds",
        default="",
        help="override the configured dataset name",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--add-name",
        "-an",
        default="",
        help="suffix appended to generated checkpoint names",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--add-name",
        "-an",
        default="",
        help="suffix appended to generated artifact names",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        default="",
        help="portable or Lightning model checkpoint to load",
    )
    return parser


def parse_train_args(argv: Sequence[str] | None = None) -> Namespace:
    return create_train_parser().parse_args(argv)


def parse_latent_args(argv: Sequence[str] | None = None) -> Namespace:
    return create_latent_parser().parse_args(argv)
