import torch
import torch.nn as nn

from ..base import BackboneBase
from ..registry import BACKBONES


@BACKBONES.register("tiny_cnn")
class TinyCNN(BackboneBase):
    """
    Very small CNN backbone for testing the framework.
    """

    def __init__(self, cfg):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=cfg.hidden_dim,
            out_channels=cfg.out_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor):

        stage1 = self.relu(self.conv1(x))
        stage2 = self.relu(self.conv2(stage1))

        return {
            "stage1": stage1,
            "stage2": stage2
        }