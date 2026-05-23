import torch

from woname.evaluators.base import MetricBase
from woname.evaluators.registry import EVALUATORS
from ..configs import DiceScoreConfig

@EVALUATORS.register("dice_score")
class DiceScore(MetricBase):
    def __init__(
            self,
            cfg: DiceScoreConfig
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
            targets * preds
        ).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            preds.sum() +
            targets.sum() +
            self.smooth
        )

        return dice 