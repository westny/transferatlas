from pathlib import Path

import pandas as pd

from analysis.plot_utils import (
    CorrelationResult,
    Metric,
    compute_correlation,
    load_evaluation_results,
    load_finetune_results,
    load_kl_matrix,
    subset_matrix,
)


def compute_forgetting_correlation(
    kl_matrix_path: Path,
    evaluation_path: Path,
    fine_tune_path: Path,
) -> CorrelationResult:
    """Compute correlation between dataset (dis)similarity and performance degradation.

    Args:
        kl_matrix_path: Path to the KL-divergence matrix CSV file.
        evaluation_path: Path to the zero-shot evaluation data CSV file.
        fine_tune_path: Path to the fine-tune evaluation data CSV file.
    Returns:
        CorrelationResult object containing the correlation coefficient and
            confidence intervals.
    """
    data_frame = combine_forgetting_data(
        kl_matrix_path,
        evaluation_path=evaluation_path,
        fine_tune_path=fine_tune_path,
    )
    dissimilarities = data_frame["kl_divergence"].tolist()
    loss_increases = data_frame["loss_increase"].tolist()
    # Compute Spearman rank correlation
    return compute_correlation(
        dissimilarities,
        loss_increases,
        method="spearman",
    )


def combine_forgetting_data(
    kl_matrix_path: Path,
    evaluation_path: Path,
    fine_tune_path: Path,
) -> pd.DataFrame:
    # Dict with keys [finetune_on][original_dataset][metric]
    finetune_data = load_finetune_results(str(fine_tune_path))
    original_datasets = set()
    for finetune_on in finetune_data:
        original_datasets.update(finetune_data[finetune_on])

    labels, dissimilarity_data = load_kl_matrix(str(kl_matrix_path))
    labels, dissimilarity_data = subset_matrix(
        labels, dissimilarity_data, list(original_datasets)
    )

    # Dict with keys [train_dataset][eval_dataset][metric]
    before_data = load_evaluation_results(str(evaluation_path))
    data: list[dict[str, float | str | int]] = []
    metric: Metric = "val_min_ade_3s"
    for dataset in labels:
        dataset_l = dataset.lower()

        for finetune_on in labels:
            finetune_on_l = finetune_on.lower()
            if finetune_on_l == dataset_l:
                continue

            backward_loss = finetune_data[finetune_on_l][dataset_l][metric]
            original_loss = before_data[dataset_l][dataset_l][metric]
            loss_increase = backward_loss - original_loss
            dissimilarity = dissimilarity_data[
                labels.index(dataset), labels.index(finetune_on)
            ]
            data.append(
                {
                    "fine_tuned_on": finetune_on_l,
                    "eval_dataset": dataset_l,
                    "org_train_dataset": dataset_l,
                    "kl_divergence": dissimilarity,
                    "loss_increase": loss_increase,
                    "metric": metric,
                }
            )

    return pd.DataFrame(data)


if __name__ == "__main__":
    kl_matrix_path = Path("results/paper/kl_divergence.csv")
    evaluation_path = Path("results/paper/zeroshot-evaluation.csv")
    fine_tune_path = Path("results/paper/finetune-evaluation.csv")
    result = compute_forgetting_correlation(
        kl_matrix_path,
        evaluation_path,
        fine_tune_path,
    )
    print(
        f"Rho: {result.rho:.4f}, CI lower: {result.ci_lower:.4f}, "
        f"CI upper: {result.ci_upper:.4f}"
    )
