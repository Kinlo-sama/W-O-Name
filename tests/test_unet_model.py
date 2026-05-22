import torch

from woname.architectures.backbones.configs import UNetEncoderConfig
from woname.architectures.decoders.configs import UNetDecoderConfig
from woname.architectures.heads.configs import SegmentationHeadConfig

from woname.vision.segmentation.configs import UNetConfig
from woname.vision.segmentation.registry import (
    SEGMENTATION_MODELS
)

import woname.architectures.backbones.models.unet_encoder
import woname.architectures.decoders.models.unet_decoder
import woname.architectures.heads.models.segmentation_head
import woname.vision.segmentation.models.unet


def main():

    cfg = UNetConfig(

        backbone=UNetEncoderConfig(),

        decoder=UNetDecoderConfig(),

        head=SegmentationHeadConfig(
            num_classes=21
        )
    )

    model = SEGMENTATION_MODELS.build(cfg)

    x = torch.randn(1, 3, 256, 256)

    y = model(x)

    print(y.shape)


if __name__ == "__main__":

    main()