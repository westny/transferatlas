import torch
from torchmetrics import Metric


class _AverageMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _accumulate(self, values: torch.Tensor) -> None:
        self.sum += values.sum()
        self.count += values.numel()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count  # type: ignore


class MinADE(_AverageMetric):
    """Average displacement error for TRACE's single predicted trajectory."""

    def update(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        valid_steps = mask.sum(dim=-1)
        scored_agents = valid_steps > 0
        displacement = torch.linalg.norm(predictions - target, dim=-1) * mask
        values = displacement[scored_agents].sum(dim=-1) / valid_steps[scored_agents]
        self._accumulate(values)


class MinFDE(_AverageMetric):
    """Final displacement error for TRACE's single predicted trajectory."""

    def update(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        sequence_length = predictions.size(1)
        batch_indices = torch.arange(predictions.size(0), device=predictions.device)
        final_indices = sequence_length - 1 - mask.flip(dims=[-1]).int().argmax(dim=-1)
        scored_agents = mask.sum(dim=-1) > 0
        final_predictions = predictions[batch_indices, final_indices][scored_agents]
        final_target = target[batch_indices, final_indices][scored_agents]
        self._accumulate(torch.linalg.norm(final_predictions - final_target, dim=-1))
