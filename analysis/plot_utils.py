from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy.stats import bootstrap

Metric = Literal[
    "val_min_fde_3s",
    "val_min_ade_3s",
    "val_loss",
    "val_mr_3s",
]

COLORS: dict[str, str] = {
    "vod": "tab:blue",
    "round": "tab:purple",
    "waymo": "tab:green",
    "ad4che": "tab:red",
    "nuscenes": "tab:orange",
    "interact": "tab:pink",
    "highd": "tab:cyan",
    "univ": "tab:brown",
    "unid": "tab:olive",
    "ind": "tab:gray",
    "apollo": "tab:cyan",
    "argoverse": "tab:purple",
    "eth": "tab:green",
    "exid": "tab:blue",
    "hotel": "tab:red",
    "i80": "tab:orange",
    "lyft": "tab:pink",
    "opendd": "tab:gray",
    "sind": "tab:brown",
    "us101": "tab:olive",
    "a43": "tab:cyan",
    "zara1": "tab:blue",
    "zara2": "tab:green",
    "default": "tab:blue",
}

METRIC_READABLE_MAP: dict[Metric, str] = {
    "val_min_fde_3s": "minFDE$_6$",
    "val_min_ade_3s": "minADE$_6$",
    "val_loss": "Validation Loss",
    "val_mr_3s": "Miss Rate",
}

DATASET_READABLE_MAP: dict[str, str] = {
    "a43": "A43",
    "vod": "View-of-Delft",
    "round": "rounD",
    "waymo": "WOMD",
    "ad4che": "AD4CHE",
    "nuscenes": "nuScenes",
    "highd": "highD",
    "univ": "univ",
    "interact": "INTERACTION",
    "unid": "uniD",
    "ind": "inD",
    "apollo": "ApolloScape",
    "argoverse": "Argoverse",
    "eth": "ETH",
    "exid": "exiD",
    "hotel": "HOTEL",
    "i80": "I-80",
    "lyft": "Lyft",
    "opendd": "OpenDD",
    "sind": "SIND",
    "us101": "US-101",
    "zara1": "ZARA1",
    "zara2": "ZARA2",
}


@dataclass
class PlotConfig:
    title: str
    x_label: str
    y_label: str
    log_x: bool = True
    log_y: bool = True


def plot_matrix(
    labels: list[str] | tuple[list[str], list[str]],
    matrix: npt.NDArray[Any],
    *,
    title: str,
    filename: Path | None = None,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "Blues",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cell_colors: dict[tuple[int, int], str] | None = None,
    transparent: bool = False,
):
    """Plot a square matrix with given labels as heatmap.

    Args:
        labels: List of dataset labels.
        matrix: Square matrix of shape (n, n).
        title: Title for the plot.
        filename: Optional filename to save (e.g. 'matrix.pdf').
        annot: Whether to annotate values in the heatmap.
        fmt: Format string for annotations.
        cmap: Colormap for heatmap.

    """
    if isinstance(labels, tuple):
        xlabels, ylabels = labels
    else:
        xlabels = ylabels = labels

    nx, ny = len(xlabels), len(ylabels)
    assert matrix.shape == (ny, nx), "Matrix shape must match labels length."

    _fig, ax = plt.subplots()
    ax = sns.heatmap(
        matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=xlabels,
        yticklabels=ylabels,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    if cell_colors is not None:
        for (i, j), color in cell_colors.items():
            ax.add_patch(
                Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.5, lw=0),
            )

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel if xlabel is not None else "Evaluation Dataset", fontsize=12)
    plt.ylabel(ylabel if ylabel is not None else "Training Dataset", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", transparent=transparent)
    else:
        plt.show()


class MetricDict(TypedDict):
    eval_dataset: str
    train_dataset: str
    val_loss: float
    val_mr_3s: float
    val_mr: float
    val_min_fde_3s: float
    val_min_fde: float
    val_min_ade_3s: float
    val_min_ade: float


class FineTuneMetricDict(TypedDict):
    eval_dataset: str
    org_dataset: str
    val_loss: float
    val_mr_3s: float
    val_mr: float
    val_min_fde_3s: float
    val_min_fde: float
    val_min_ade_3s: float
    val_min_ade: float


def load_evaluation_results(
    csv_file: str,
) -> dict[str, dict[str, MetricDict]]:
    df = pd.read_csv(csv_file)
    data: dict[str, dict[str, MetricDict]] = {}
    for _, row in df.iterrows():
        metric = MetricDict(**row.to_dict())
        train_dataset = metric["train_dataset"].lower()
        eval_dataset = metric["eval_dataset"].lower()
        if train_dataset not in data:
            data[train_dataset] = {}
        data[train_dataset][eval_dataset] = metric
    return data


def load_finetune_results(
    csv_file: str,
) -> dict[str, dict[str, FineTuneMetricDict]]:
    df = pd.read_csv(csv_file)
    data: dict[str, dict[str, FineTuneMetricDict]] = {}
    for _, row in df.iterrows():
        metric = FineTuneMetricDict(**row.to_dict())
        finetune_dataset = metric["org_dataset"].lower()
        eval_dataset = metric["eval_dataset"].lower()
        if finetune_dataset not in data:
            data[finetune_dataset] = {}
        data[finetune_dataset][eval_dataset] = metric
    return data


def load_kl_matrix(csv_file: str) -> tuple[list[str], npt.NDArray[np.float64]]:
    df = pd.read_csv(csv_file, index_col=0)
    labels = [str(label).lower() for label in df.columns]
    row_labels = [str(label).lower() for label in df.index]
    if row_labels != labels:
        raise ValueError(
            "KL matrix row and column labels must match in the same order."
        )
    return labels, df.to_numpy(dtype=np.float64)


def verify_evaluation(
    data: dict[str, dict[str, MetricDict]],
    labels: list[str],
    matrix: npt.NDArray[np.float64],
    metric_name: str = "val_min_fde_3s",
) -> npt.NDArray[np.int64]:
    n = len(labels)
    if matrix.shape != (n, n):
        msg = f"Matrix shape {matrix.shape} does not match labels length {n}."
        raise ValueError(msg)

    verification_matrix = np.full((n, n), -1, dtype=np.int64)  # -1 = missing

    for i, train_label in enumerate(labels):
        # KL scores for fixed training dataset vs all eval datasets (lower is better)
        kl_scores = matrix[:, i].astype(np.float64)

        # Metrics aligned to labels; missing evals set to inf so they sort to the end
        present = np.array(
            [eval_label in data.get(train_label, {}) for eval_label in labels],
            dtype=bool,
        )
        metrics_vec = np.full(n, np.inf, dtype=np.float64)
        for j, eval_label in enumerate(labels):
            if present[j]:
                metrics_vec[j] = data[train_label][eval_label][metric_name]

        order_kl = np.argsort(kl_scores, kind="stable")
        rank_kl = np.empty(n, dtype=np.int64)
        rank_kl[order_kl] = np.arange(n, dtype=np.int64)

        order_metrics = np.argsort(metrics_vec, kind="stable")
        rank_metrics = np.empty(n, dtype=np.int64)
        rank_metrics[order_metrics] = np.arange(n, dtype=np.int64)

        # Absolute rank difference per eval dataset; mask out missing metrics
        diff = np.abs(rank_metrics - rank_kl).astype(np.int64)
        diff[~present] = -1  # mark missing

        verification_matrix[i, :] = diff

    return verification_matrix


def subset_matrix(
    labels: list[str],
    matrix: npt.NDArray[np.float64],
    keep: list[str],
) -> tuple[list[str], npt.NDArray[np.float64]]:
    # Map labels to indices
    label_to_idx = {label: i for i, label in enumerate(labels)}
    indices = [label_to_idx[label] for label in keep if label in label_to_idx]
    new_labels = [labels[i] for i in indices]
    new_matrix = matrix[np.ix_(indices, indices)]

    return new_labels, new_matrix


@dataclass
class CorrelationResult:
    rho: float
    p_value: float
    ci_lower: float
    ci_upper: float


def _to_correlation_result(
    corr_df: pd.DataFrame,
) -> CorrelationResult:
    ci_l, ci_h = corr_df["CI95"].values[0]
    return CorrelationResult(
        rho=corr_df["r"].values[0],
        p_value=corr_df["p_val"].values[0],
        ci_lower=ci_l,
        ci_upper=ci_h,
    )


CorrelationMethod = Literal[
    "spearman",
    "pearson",
    "kendall",
    "bicor",
    "shepherd",
    "percbend",
    "skipped",
    "distance",
]


def compute_correlation(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    method: CorrelationMethod = "spearman",
    n_boot: int | None = 1000,
) -> CorrelationResult:
    if n_boot is not None:
        return bootstrap_correlation(x, y, method=method, n_boot=n_boot)

    if method == "distance":
        distance_correlation, p_value = pg.distance_corr(
            x=x, y=y, n_boot=n_boot or 1000
        )
        return CorrelationResult(
            rho=distance_correlation,
            p_value=p_value,
            ci_lower=np.nan,
            ci_upper=np.nan,
        )

    return _to_correlation_result(pg.corr(x=x, y=y, method=method))  # pyright: ignore[reportArgumentType]


def bootstrap_correlation(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    method: CorrelationMethod = "spearman",
    n_boot: int = 10000,
    ci_level: float = 0.95,
) -> CorrelationResult:
    def corr_func(x, y) -> float:
        corr_res = compute_correlation(x, y, method=method, n_boot=None)
        return corr_res.rho

    res = bootstrap(
        data=(np.array(x), np.array(y)),
        statistic=corr_func,
        vectorized=False,
        n_resamples=n_boot,
        confidence_level=ci_level,
        paired=True,
        method="BCa",
    )
    ci_lower, ci_upper = res.confidence_interval
    result = compute_correlation(x, y, method=method, n_boot=None)
    return CorrelationResult(
        rho=result.rho,
        p_value=result.p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def plot_correlation(
    df: pd.DataFrame,
    groups: dict[str, set[str]] | None = None,
    title: str | None = None,
    save_path: Path | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Axes:
    # --- Group assignment ----------------------------------------------------
    df["group"] = "Other"
    if groups is not None:
        for name, members in groups.items():
            df.loc[
                (df["kl_divergence"] > 0)
                & (df["eval_value"] > 0)
                & df["model_train_dataset"].isin(list(members)),
                "group",
            ] = name

    fig, ax = plt.subplots(figsize=(3.25, 2.4375))

    # --- Theme ---------------------------------------------------------------
    sns.set_theme(
        style="white",
        font_scale=1.3,
        context="paper",
        font="serif",
        rc={
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        },
    )

    hue = "group" if groups is not None else None
    sns.scatterplot(
        data=df,
        x="kl_divergence",
        y="eval_value",
        alpha=0.65,
        edgecolors="k",
        linewidths=0.5,
        s=15,
        hue=hue,
        style=hue,
        ax=ax,
    )

    # --- Log scales ----------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    if x_label is not None:
        ax.set_xlabel(x_label + " (log scale)", fontsize=8)
    if y_label is not None:
        ax.set_ylabel(y_label + " (log scale)", fontsize=8)

    if title:
        ax.set_title(title, fontsize=10, pad=5)

    # --- Grid ----------------------------------------------------------------
    ax.grid(True, which="major", ls="--", lw=0.75, alpha=0.6)

    # --- Legend --------------------------------------------------------------
    if groups is not None:
        ax.legend(
            loc="best",
            fontsize=7,
            frameon=False,
        )

    # --- Arrowheads ----------------------------------------------------------
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.annotate(
        "",
        xy=(x_max, y_min),
        xytext=(x_min, y_min),
        arrowprops={
            "arrowstyle": "-|>,head_width=0.15,head_length=0.25",
            "lw": 1.0,
            "color": "black",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )

    ax.annotate(
        "",
        xy=(x_min, y_max),
        xytext=(x_min, y_min),
        arrowprops={
            "arrowstyle": "-|>,head_width=0.15,head_length=0.25",
            "lw": 1.0,
            "color": "black",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )

    sns.despine(ax=ax, left=True, bottom=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:g}"))
    ax.tick_params(axis="both", which="both", labelsize=7, length=0, width=0)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
    return ax
