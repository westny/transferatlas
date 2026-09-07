"""Analyze how dataset divergence relates to zero-shot transfer performance."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import pandas as pd

from analysis.plot_utils import (
    METRIC_READABLE_MAP,
    CorrelationMethod,
    CorrelationResult,
    Metric,
    compute_correlation,
    load_evaluation_results,
    load_kl_matrix,
    plot_correlation,
)

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman"]


def _identity(value: float) -> float:
    return value


def compute_zeroshot_correlation(
    kl_matrix_path: Path,
    eval_data_path: Path,
    metric: Metric,
    kl_transform: Callable[[float], float] = _identity,
    ignore_train_datasets: set[str] | None = None,
    ignore_eval_datasets: set[str] | None = None,
    method: CorrelationMethod = "spearman",
    plot: bool = True,
    save_path: Path | None = None,
    title: str | None = None,
    groups: dict[str, set[str]] | None = None,
    n_boot: int | None = None,
) -> CorrelationResult:
    """Compute correlation between (dis)similarity between datasets and zero-shot.

    Args:
        kl_matrix_path: KL-divergence matrix CSV path.
        eval_data_path: zero-shot evaluation data path (csv file).
        metric: what metric to use for evaluation (e.g., "val_min_ade_3s").
        kl_transform: function applied to KL-divergence values. Defaults to identity.
        ignore_train_datasets: set of training datasets to ignore.
        ignore_eval_datasets: set of evaluation datasets to ignore.
        method: correlation method to use. Defaults to "spearman".
        plot: if correlation plot should be generated. Defaults to True.
        save_path: save path for the plot. Defaults to None.
        title: plot title. Defaults to None.
        groups: grouping of datasets for plotting. Defaults to None.
        n_boot: number of bootstrap samples for confidence intervals. Defaults to None
            (default CI computation, if available, depending on the correlation method).

    Returns:
        CorrelationResult: correlation result object.
    """
    ignore_train_datasets = ignore_train_datasets or set()
    ignore_eval_datasets = ignore_eval_datasets or set()

    data_frame = combine_kl_and_evaluation(
        kl_matrix_path,
        eval_data_path,
        ignore_train_datasets=ignore_train_datasets,
        ignore_eval_datasets=ignore_eval_datasets,
        metric=metric,
        kl_transform=kl_transform,
    )

    overall_correlation = compute_correlation(
        data_frame["kl_divergence"].tolist(),
        data_frame["eval_value"].tolist(),
        method=method,
        n_boot=n_boot,
    )
    if plot:
        _ = plot_correlation(
            df=data_frame,
            groups=groups,
            title=title,
            save_path=save_path,
            x_label=r"$D_{\mathrm{KL}}(\mathcal{D}_e \| \mathcal{D}_t)$",
            y_label=METRIC_READABLE_MAP.get(metric, metric),
        )

    return overall_correlation


def extract_kl_data(
    kl_matrix_path: Path,
    ignore_train_datasets: set[str] | None = None,
    ignore_eval_datasets: set[str] | None = None,
    kl_transform: Callable[[float], float] = _identity,
) -> pd.DataFrame:
    labels, matrix = load_kl_matrix(str(kl_matrix_path))

    # Zero shot correlation with KL divergence
    ignore_train_datasets = ignore_train_datasets or set()
    ignore_eval_datasets = ignore_eval_datasets or set()

    data_dict: list[dict[str, float | str]] = []
    for i, train_label in enumerate(labels):
        train_label_l = train_label.lower()
        if train_label_l in ignore_train_datasets:
            continue

        for j, eval_label in enumerate(labels):
            eval_label_l = eval_label.lower()
            if eval_label_l == train_label_l or eval_label_l in ignore_eval_datasets:
                continue

            divergence = kl_transform(matrix[j, i])
            data_dict.append(
                {
                    "model_train_dataset": train_label_l,
                    "eval_dataset": eval_label_l,
                    "kl_divergence": divergence,
                }
            )

    return pd.DataFrame(data_dict)


def combine_kl_and_evaluation(
    kl_matrix_path: Path,
    eval_data_path: Path,
    ignore_train_datasets: set[str] | None = None,
    ignore_eval_datasets: set[str] | None = None,
    metric: Metric = "val_min_ade_3s",
    kl_transform: Callable[[float], float] = _identity,
) -> pd.DataFrame:
    labels, matrix = load_kl_matrix(str(kl_matrix_path))
    eval_data = load_evaluation_results(str(eval_data_path))

    # Zero shot correlation with KL divergence
    ignore_train_datasets = ignore_train_datasets or set()
    ignore_eval_datasets = ignore_eval_datasets or set()

    data_dict: list[dict[str, float | str]] = []
    for i, train_label in enumerate(labels):
        train_label_l = train_label.lower()
        if train_label_l not in eval_data or train_label_l in ignore_train_datasets:
            continue

        for j, eval_label in enumerate(labels):
            eval_label_l = eval_label.lower()
            if (
                eval_label_l == train_label_l
                or eval_label_l not in eval_data[train_label_l]
                or eval_label_l in ignore_eval_datasets
            ):
                continue

            divergence = kl_transform(matrix[j, i])
            val = eval_data[train_label_l][eval_label_l][metric]

            data_dict.append(
                {
                    "model_train_dataset": train_label_l,
                    "eval_dataset": eval_label_l,
                    "kl_divergence": divergence,
                    "eval_value": val,
                }
            )

    return pd.DataFrame(data_dict)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Reproduce the paper's zero-shot transfer correlation."
    )
    _ = parser.add_argument(
        "--plot",
        action="store_true",
        help="display the correlation plot (requires a graphical backend and LaTeX)",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    eval_data_path = Path("results/paper/zeroshot-evaluation.csv")
    kl_matrix_path = Path("results/paper/kl_divergence.csv")
    metric: Metric = "val_min_ade_3s"
    skip: set[str] | None = None

    result = compute_zeroshot_correlation(
        kl_matrix_path=kl_matrix_path,
        eval_data_path=eval_data_path,
        metric=metric,
        n_boot=None,
        ignore_eval_datasets=skip,
        ignore_train_datasets=skip,
        plot=args.plot,
    )
    print(f"Overall correlation: {result.rho:.4f} (p={result.p_value:.4f})")


if __name__ == "__main__":
    main()
