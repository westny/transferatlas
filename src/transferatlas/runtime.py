from __future__ import annotations

import shutil
from argparse import Namespace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import torch
import yaml
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy


def require_graph_backend() -> None:
    """Fail early when no compatible PyG radius backend is installed."""
    try:
        pyg_lib_version = version("pyg-lib")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Training and latent extraction require a graph backend. Install one "
            "with `uv sync --extra cpu` or `uv sync --extra cu126`."
        ) from error

    try:
        radius_op = torch.ops.pyg.radius
    except AttributeError as error:
        raise RuntimeError(
            f"pyg-lib {pyg_lib_version} does not provide the radius operator for "
            f"PyTorch {torch.__version__}. Recreate the environment with "
            "`uv sync --extra cpu --reinstall` or "
            "`uv sync --extra cu126 --reinstall`."
        ) from error

    if radius_op is None:
        raise RuntimeError("The installed pyg-lib radius operator is unavailable.")


def create_experiment_logger(
    *,
    dry_run: bool,
    use_logger: bool,
    project_name: str,
    run_name: str,
) -> Logger | None:
    """Based on the provided arguments, return a WandbLogger or None."""
    if dry_run or not use_logger:
        return None
    return WandbLogger(project=project_name, name=run_name)


def configure_trainer_hardware(
    args: Namespace,
) -> tuple[int, Literal["auto"] | DDPStrategy, str]:
    """Setup CUDA devices, strategy and accelerator.

    Returns:
        A tuple containing the number of devices, strategy and accelerator.
    """
    # Determine the number of devices, strategy and accelerator
    if torch.cuda.is_available() and args.use_cuda:
        torch.set_float32_matmul_precision("medium")
        devices = -1 if torch.cuda.device_count() > 1 else 1
        strategy = (
            DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            if devices == -1
            else "auto"
        )
        return devices, strategy, "auto"

    return 1, "auto", "cpu"


def load_config(config: str | Path) -> dict:
    """Load a YAML configuration by exact name or filesystem path."""
    config_path = resolve_config_path(config)
    with config_path.open() as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def resolve_config_path(config: str | Path) -> Path:
    """Resolve a configuration path without ambiguous substring matching."""
    requested = Path(config).expanduser()
    if requested.is_file():
        return requested

    filename = requested.name
    if requested.suffix not in {".yml", ".yaml"}:
        filename = f"{filename}.yml"

    config_root = Path("configs")
    matches = sorted(config_root.rglob(filename)) if config_root.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"Configuration not found: {config}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise ValueError(f"Configuration name is ambiguous: {paths}")
    return matches[0]


def copy_config_file(config: str | Path, destination: Path) -> None:
    """Copy the configuration file to the destination directory.

    Args:
        config: the name of the config file (without extension).
        destination: directory where the configuration should be copied.

    """
    config_path = resolve_config_path(config)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, destination / "used_config.yml")
