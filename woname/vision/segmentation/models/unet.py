import torch

from woname.architectures.backbones.registry import BACKBONES
from woname.architectures.decoders.registry import DECODERS
from woname.architectures.heads.registry import HEADS

from ..base import SegmentationModelBase
from ..registry import SEGMENTATION_MODELS
from ..configs import UNetConfig

@SEGMENTATION_MODELS.register("unet")
class UNet(SegmentationModelBase):
    def __init__(
            self,
            cfg: UNetConfig
    ):
        super().__init__()
        self.backbone = BACKBONES.build(
            cfg.backbone
        )
        self.decoder = DECODERS.build(
            cfg.decoder,
            encoder_channels=self.backbone.out_channels
        )
        self.head = HEADS.build(
            cfg.head,
            in_channels=self.decoder.out_channels
        )
    
    def forward(
            self, 
            x: torch.Tensor
            ):
        features = self.backbone(x)
        x = self.decoder(features)
        logits = self.head(x)
        return logits

