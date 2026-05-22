from abc import ABC, abstractmethod
from typing import Dict

import torch 
import torch.nn as nn

class DecoderBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass
