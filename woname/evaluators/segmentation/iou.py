import torch

from woname.evaluators.base import MetricBase
from woname.evaluators.registry import EVALUATORS
from ..configs import IoUConfig

@EVALUATORS.register("iou", IoUConfig)
class IoU(MetricBase):
    def __init__(
            self,
            cfg: IoUConfig
    ):
        super().__init__()
        self.threshold = cfg.threshold
        self.smooth = cfg.smooth

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (
            preds * targets
        ).sum()

        union = (
            preds.sum() + 
            targets.sum() -
            intersection
        )

        iou = (
            intersection + self.smooth
        ) / (
            union + self.smooth
        )
        
        return iou