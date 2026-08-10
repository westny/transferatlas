from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


@dataclass(frozen=True)
class GaussianStatistics:
    mean: torch.Tensor
    covariance: torch.Tensor


def _validate_statistics(statistics: GaussianStatistics, name: str) -> None:
    mean = statistics.mean
    covariance = statistics.covariance

    if mean.ndim != 1:
        raise ValueError(f"{name} mean must be one-dimensional, got {mean.shape}.")
    if covariance.shape != (mean.numel(), mean.numel()):
        raise ValueError(
            f"{name} covariance must have shape ({mean.numel()}, {mean.numel()}), "
            f"got {tuple(covariance.shape)}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise ValueError(f"{name} statistics contain non-finite values.")
    if not torch.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} covariance is not symmetric.")

    try:
        torch.linalg.cholesky(covariance)
    except torch.linalg.LinAlgError as error:
        raise ValueError(f"{name} covariance is not positive definite.") from error


def load_statistics(dataset_dir: str | Path) -> GaussianStatistics:
    dataset_dir = Path(dataset_dir)
    mean_path = dataset_dir / "mean.pt"
    covariance_path = dataset_dir / "covariance.pt"
    if not mean_path.is_file() or not covariance_path.is_file():
        raise FileNotFoundError(f"Expected mean.pt and covariance.pt in {dataset_dir}.")

    mean = torch.load(mean_path, map_location="cpu", weights_only=True)
    covariance = torch.load(covariance_path, map_location="cpu", weights_only=True)
    if not isinstance(mean, torch.Tensor) or not isinstance(covariance, torch.Tensor):
        raise TypeError(f"Statistics in {dataset_dir} must be PyTorch tensors.")

    statistics = GaussianStatistics(mean=mean.double(), covariance=covariance.double())
    _validate_statistics(statistics, dataset_dir.name)
    return statistics


def gaussian_kl_divergence(
    source: GaussianStatistics,
    target: GaussianStatistics,
) -> torch.Tensor:
    """Compute KL(source || target) for two multivariate Gaussians."""
    _validate_statistics(source, "source")
    _validate_statistics(target, "target")
    if source.mean.shape != target.mean.shape:
        raise ValueError(
            "Source and target dimensions differ: "
            f"{source.mean.numel()} != {target.mean.numel()}."
        )

    target_cholesky = torch.linalg.cholesky(target.covariance)
    trace_term = torch.trace(torch.cholesky_solve(source.covariance, target_cholesky))
    mean_delta = (target.mean - source.mean).unsqueeze(1)
    mahalanobis = (
        mean_delta.T @ torch.cholesky_solve(mean_delta, target_cholesky)
    ).squeeze()

    source_cholesky = torch.linalg.cholesky(source.covariance)
    source_logdet = 2 * torch.log(torch.diagonal(source_cholesky)).sum()
    target_logdet = 2 * torch.log(torch.diagonal(target_cholesky)).sum()
    dimension = source.mean.numel()
    divergence = 0.5 * (
        target_logdet - source_logdet - dimension + trace_term + mahalanobis
    )
    return divergence.clamp_min(0.0)


def discover_datasets(statistics_dir: str | Path) -> list[str]:
    statistics_dir = Path(statistics_dir)
    if not statistics_dir.is_dir():
        raise NotADirectoryError(
            f"Statistics directory does not exist: {statistics_dir}"
        )

    datasets = sorted(
        (
            path.name
            for path in statistics_dir.iterdir()
            if path.is_dir()
            and (path / "mean.pt").is_file()
            and (path / "covariance.pt").is_file()
        ),
        key=str.casefold,
    )
    if not datasets:
        raise FileNotFoundError(f"No dataset statistics found in {statistics_dir}.")
    return datasets


def load_all_statistics(
    statistics_dir: str | Path,
    datasets: Iterable[str] | None = None,
) -> dict[str, GaussianStatistics]:
    statistics_dir = Path(statistics_dir)
    names = (
        list(datasets) if datasets is not None else discover_datasets(statistics_dir)
    )
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique.")
    return {name: load_statistics(statistics_dir / name) for name in names}


def compute_kl_matrix(
    statistics: Mapping[str, GaussianStatistics],
) -> pd.DataFrame:
    """Return a matrix whose (i, j) entry is KL(dataset_i || dataset_j)."""
    names = list(statistics)
    if not names:
        raise ValueError("At least one dataset is required.")

    matrix = torch.zeros((len(names), len(names)), dtype=torch.float64)
    for source_index, source_name in enumerate(names):
        for target_index, target_name in enumerate(names):
            if source_index == target_index:
                continue
            matrix[source_index, target_index] = gaussian_kl_divergence(
                statistics[source_name], statistics[target_name]
            )

    frame = pd.DataFrame(
        matrix.numpy(),
        index=pd.Index(data=names),
        columns=pd.Index(data=names),
    )
    frame.index.name = "dataset"
    return frame


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Compute directed pairwise KL divergence between latent Gaussians."
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="Directory containing one mean.pt/covariance.pt directory per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <stats-dir>/kl_divergence.csv).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset names and output order (default: discovered alphabetically).",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output = args.output or args.stats_dir / "kl_divergence.csv"
    statistics = load_all_statistics(args.stats_dir, args.datasets)
    matrix = compute_kl_matrix(statistics)

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output)
    print(f"Saved {len(matrix)}x{len(matrix)} directed KL matrix to {output}")


if __name__ == "__main__":
    main()
