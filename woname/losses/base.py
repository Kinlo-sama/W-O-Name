from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class LossBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        pass