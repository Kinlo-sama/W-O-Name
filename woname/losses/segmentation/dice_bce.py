import torch

from woname.losses.base import LossBase
from woname.losses.registry import LOSSES
from woname.losses.configs import DiceBCELossConfig

from .dice import DiceLoss
from .bce import BCELoss

@LOSSES.register("dice_bce_loss", DiceBCELossConfig)
class DiceBCELoss(LossBase):
    def __init__(
            self, 
            cfg: DiceBCELossConfig
            ):
        super().__init__()
        
        self.dice_loss = DiceLoss(cfg.dice_cfg)
        self.bce_loss = BCELoss(cfg.bce_cfg)

        self.dice_weight = cfg.dice_weight
        self.bce_weight = cfg.bce_weight
    
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