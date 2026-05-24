import torch
import torch.nn as nn

from ..base import HeadBase
from ..configs import SegmentationHeadConfig
from ..registry import HEADS

@HEADS.register("segmentation_head", SegmentationHeadConfig)
class SegmentationHead(HeadBase):
    def __init__(
            self,
            cfg: SegmentationHeadConfig,
            in_channels: int
    ):
        super().__init__()
        self.classifier = nn.Conv2d(
            in_channels,
            cfg.num_classes,
            kernel_size=1,
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(x)
    
