from woname.losses.base import LossBase
from woname.losses.registry import LOSSES
import torch.nn as nn
import torch

@LOSSES.register("bce_loss")
class BCELoss(LossBase):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
            self, 
            logits: torch.Tensor, 
            targets: torch.Tensor
        ) -> torch.Tensor:
        targets = targets.float()

        return self.loss_fn(logits, targets)