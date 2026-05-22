import torch

from woname.architectures.backbones.configs import UNetEncoderConfig
from woname.architectures.backbones.registry import BACKBONES

from woname.architectures.decoders.configs import UNetDecoderConfig
from woname.architectures.decoders.registry import DECODERS

import woname.architectures.backbones.models.unet_encoder
import woname.architectures.decoders.models.unet_decoder


def main():

    backbone_cfg = UNetEncoderConfig()

    backbone = BACKBONES.build(backbone_cfg)

    decoder_cfg = UNetDecoderConfig()

    decoder = DECODERS.build(
        decoder_cfg,
        encoder_channels=backbone.out_channels
    )

    x = torch.randn(1, 3, 256, 256)

    features = backbone(x)

    y = decoder(features)

    print(y.shape)


if __name__ == "__main__":

    main()