import torch

from woname.evaluators.base import MetricBase
from woname.evaluators.registry import EVALUATORS
from ..configs import PixelAccuracyConfig

@EVALUATORS.register("pixel_accuracy")
class PixelAccuracy(MetricBase):
    def __init__(
            self,
            cfg: PixelAccuracyConfig
    ):
        super().__init__()
        self.threshold = cfg.threshold

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        preds = (
            preds > self.threshold
        ).float()

        correct = (
            preds == targets
        ).float()

        return correct.mean()