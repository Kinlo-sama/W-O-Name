from abc import ABC, abstractmethod
from woname.architectures.backbones.registry import BACKBONES
from woname.architectures.necks.registry import NECKS
from woname.architectures.decoders.registry import DECODERS
from woname.architectures.heads.registry import HEADS
import torch
import torch.nn as nn


class SegmentationModelBase(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BACKBONES.build(cfg.backbone)

        if cfg.neck is not None:
            self.neck = NECKS.build(cfg.neck)
        
        encoder_channels = self.neck.out_channels if hasattr(self, 'neck') else self.backbone.out_channels

        if cfg.decoder is not None:
            self.decoder = DECODERS.build(
                cfg.decoder,
                encoder_channels=encoder_channels
            )

        if hasattr(self, "decoder"):
            in_channels = self.decoder.out_channels
        elif hasattr(self, "neck"):
            in_channels = self.neck.out_channels
        else:
            in_channels = self.backbone.out_channels

        self.head = HEADS.build(
            cfg.head,
            in_channels=in_channels
            )
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor
        ) -> torch.Tensor:
        pass