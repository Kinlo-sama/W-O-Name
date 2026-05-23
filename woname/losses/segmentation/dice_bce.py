import torch

from woname.losses.base import LossBase
from woname.losses.registry import LOSSES

from .dice import DiceLoss
from .bce import BCELoss

@LOSSES.register("dice_bce_loss")
class DiceBCELoss(LossBase):
    def __init__(
            self, 
            dice_weight: float = 1.0,
            bce_weight: float = 1.0
            ):
        super().__init__()
        
        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        dice = self.dice_loss(
            logits,
            targets
        )
        bce = self.bce_loss(
            logits,
            targets
        )
        return (
            self.dice_weight * dice +
            self.bce_weight * bce
        )