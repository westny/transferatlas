from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
from lightning.pytorch import seed_everything
from torch.multiprocessing import set_sharing_strategy
from tqdm import tqdm

from transferatlas.cli.arguments import LatentArgs, parse_latent_args
from transferatlas.config import ExperimentConfig
from transferatlas.data.datamodule import TrajectoryDataModule
from transferatlas.data.dataset import dataset_group_name
from transferatlas.models.trace.lightning import TraceLightningModule
from transferatlas.models.trace.model import TraceModel
from transferatlas.runtime import load_config, require_graph_backend

logging.basicConfig(level=logging.INFO, format="%(message)s")
printer = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument*",
)
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

set_sharing_strategy("file_system")

LOWRANK_RANK = 16
JITTER_ALPHA = 1e-4

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------


def resolve_checkpoint(
    checkpoint_path: str,
    dry_run: bool,
) -> Path | None:
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    if dry_run:
        return None
    raise ValueError("--checkpoint is required when --dry-run is false.")


def symmetrize(cov: torch.Tensor) -> torch.Tensor:
    return 0.5 * (cov + cov.T)


def stable_scale(cov: torch.Tensor) -> torch.Tensor:
    """
    Scale used for adaptive jitter:

        scale = max(trace(cov) / D, 1e-12)
    """
    cov = symmetrize(cov)

    dim = cov.shape[0]
    scale = torch.trace(cov) / dim

    return torch.clamp(
        scale,
        min=torch.tensor(
            1e-12,
            device=cov.device,
            dtype=cov.dtype,
        ),
    )


def lowrank_cov_jitter(
    cov: torch.Tensor,
    rank: int,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    """
    Low-rank covariance with isotropic jitter:

        Sigma_r = U_r Lambda_r U_r^T + eps I

    where U_r and Lambda_r are taken from the top-r eigendirections of cov.
    """
    cov = symmetrize(cov)

    dim = cov.shape[0]

    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")

    eps = alpha * stable_scale(cov)
    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)

    if rank == 0:
        cov_reg = eps * eye
        return symmetrize(cov_reg), float(eps.item())

    rank = min(rank, dim)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_r = torch.clamp(eigvals[:rank], min=0.0)
    eigvecs_r = eigvecs[:, :rank]

    cov_lowrank = eigvecs_r @ torch.diag(eigvals_r) @ eigvecs_r.T
    cov_reg = cov_lowrank + eps * eye

    return symmetrize(cov_reg), float(eps.item())


def update_running_mean_cov(
    n_old: int,
    mean_old: torch.Tensor | None,
    m2_old: torch.Tensor | None,
    x: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Exact batch update for running mean and unnormalized covariance.

    M2 stores:

        M2 = sum_i (x_i - mean)(x_i - mean)^T
    """
    b = x.size(0)

    if b == 0:
        if mean_old is None or m2_old is None:
            raise ValueError("Received empty first batch.")
        return n_old, mean_old, m2_old

    mean_batch = x.mean(dim=0)
    centered = x - mean_batch
    M2_batch = centered.T @ centered

    if n_old == 0:
        return b, mean_batch, M2_batch

    if mean_old is None or m2_old is None:
        raise ValueError("mean_old and m2_old must be initialized when n_old > 0.")

    n_new = n_old + b
    delta = mean_batch - mean_old

    mean_new = mean_old + delta * (b / n_new)

    m2_new = m2_old + M2_batch + torch.outer(delta, delta) * (n_old * b / n_new)

    return n_new, mean_new, m2_new


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def compute(
    args: LatentArgs,
    config: ExperimentConfig,
    save_name: str,
) -> None:
    require_graph_backend()
    checkpoint = resolve_checkpoint(
        checkpoint_path=args.checkpoint,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        args.small_ds = True

    # ---------------------------------------------------------------------
    # Setup model
    # ---------------------------------------------------------------------

    if checkpoint is not None:
        printer.info(f"Loading checkpoint: {checkpoint}")
        net = TraceModel.from_checkpoint(checkpoint, config["model"])
    else:
        net = TraceModel(config["model"])
    model = TraceLightningModule(net, config["training"])

    # ---------------------------------------------------------------------
    # Setup datamodule
    # ---------------------------------------------------------------------

    datamodule = TrajectoryDataModule(config["datamodule"], args)
    datamodule.setup()

    # Intentionally use validation loader for dataset statistics.
    stats_loader = datamodule.val_dataloader()

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------

    stats_dataset = dataset_group_name(config["datamodule"]["name"])
    output_name = (
        checkpoint.stem if args.checkpoint and checkpoint is not None else save_name
    )

    save_dir = Path("artifacts/latent-statistics") / output_name / stats_dataset

    # ---------------------------------------------------------------------
    # Running statistics
    # ---------------------------------------------------------------------

    n_scenes = 0

    mean_mu: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    cov_within_sum: torch.Tensor | None = None

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )

    _ = model.to(device)
    _ = model.eval()

    # ---------------------------------------------------------------------
    # Process batches
    # ---------------------------------------------------------------------

    with torch.inference_mode():
        for batch in tqdm(
            stats_loader,
            desc="Computing scene Gaussian statistics",
            total=len(stats_loader),
        ):
            batch = batch.to(device)
            stats = model.compute_scene_statistics(batch)

            scene_mean = stats["mean"].double()
            scene_covariance = stats["covariance"].double()

            if scene_mean.ndim != 2:
                raise RuntimeError(
                    f"Expected scene mean shape [B, D], got {tuple(scene_mean.shape)}"
                )

            if scene_covariance.ndim != 3:
                raise RuntimeError(
                    "Expected scene covariance shape [B, D, D], got "
                    + f"{tuple(scene_covariance.shape)}"
                )

            if scene_covariance.size(1) != scene_mean.size(1) or scene_covariance.size(
                2
            ) != scene_mean.size(1):
                raise RuntimeError(
                    "Scene covariance shape is incompatible with scene mean: "
                    + f"mean={tuple(scene_mean.shape)}, "
                    + f"covariance={tuple(scene_covariance.shape)}"
                )

            if cov_within_sum is None:
                cov_within_sum = torch.zeros_like(scene_covariance[0])

            n_scenes, mean_mu, m2 = update_running_mean_cov(
                n_scenes,
                mean_mu,
                m2,
                scene_mean,
            )

            cov_within_sum += scene_covariance.sum(dim=0)

    if n_scenes == 0:
        raise RuntimeError("No scenes were processed; cannot compute statistics.")

    if mean_mu is None or m2 is None or cov_within_sum is None:
        raise RuntimeError("Statistics were not initialized correctly.")

    # ---------------------------------------------------------------------
    # Estimate the full latent covariance
    # ---------------------------------------------------------------------

    mean = mean_mu

    cov_between = symmetrize(m2 / n_scenes)
    cov_within = symmetrize(cov_within_sum / n_scenes)
    covariance, eps = lowrank_cov_jitter(
        cov_between + cov_within,
        rank=LOWRANK_RANK,
        alpha=JITTER_ALPHA,
    )

    dim = mean.numel()

    # ---------------------------------------------------------------------
    # Minimal logging
    # ---------------------------------------------------------------------

    printer.info("---- dataset statistics ----")
    printer.info(f"n_scenes: {n_scenes}")
    printer.info(f"embedding_dim: {dim}")
    printer.info(f"covariance_rank: {min(LOWRANK_RANK, dim)}")
    printer.info(f"covariance_jitter: {eps:.6g}")

    # ---------------------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------------------

    if not args.dry_run:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mean.cpu(), save_dir / "mean.pt")
        torch.save(covariance.cpu(), save_dir / "covariance.pt")

        printer.info(f"Saved dataset statistics to {save_dir}")

    else:
        printer.info("Dry run mode: not saving files.")
        printer.info(f"Mean shape: {tuple(mean.shape)}")
        printer.info(f"Covariance shape: {tuple(covariance.shape)}")

    printer.info("Done.")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_latent_args()
    _ = seed_everything(args.seed, workers=True)

    config = load_config(args.config)
    if args.dataset:
        config["datamodule"]["name"] = args.dataset
    if args.root:
        config["datamodule"]["root"] = args.root

    model_name = config["model"]["name"]
    ds_name = dataset_group_name(config["datamodule"]["name"])
    config["training"]["dataset"] = ds_name

    add_name = f"-{args.add_name}" if args.add_name else ""
    full_save_name = f"{model_name}{add_name}-{ds_name}"

    compute(
        args,
        config,
        full_save_name,
    )


if __name__ == "__main__":
    main()
