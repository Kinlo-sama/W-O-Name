# backbone/base.py

from abc import ABC, abstractmethod
from typing import Dict
import torch 
import torch.nn as nn

class BackboneBase(nn.Module, ABC):
    """
    Base class for all backbone in the framework.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args: 
            x: input tensor [B, C, H, W]

        Returns:
            dict of features maps (multi-scale or single)
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def out_channels(self) -> dict[str, int]:
        pass

    def init_weights(self):
        """
        Initialize model weights.
        Override in subclasses if needed.
        """
        pass

    def freeze_stage(self):
        pass