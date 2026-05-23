import torch
import torch.nn as nn

from woname.losses.base import LossBase
from woname.losses.registry import LOSSES

@LOSSES.register("dice_loss")
class DiceLoss(LossBase):
    def __init__(
            self,
            smooth: float = 1e-6
    ):
        super().__init__()
        self.smooth = smooth
    
    def forward(
            self,
            logits:  torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum() +
            targets.sum() +
            self.smooth
        )
        return 1.0 - dice
    