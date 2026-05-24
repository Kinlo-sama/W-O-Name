import torch
import torch.nn as nn

from woname.layers.decoder_blocks import UNetDecoderBlock

from ..registry import DECODERS
from ..configs import UNetDecoderConfig
from ..base import DecoderBase

@DECODERS.register("unet_decoder", UNetDecoderConfig)
class UNetDecoder(DecoderBase):
    def __init__(self,
                 cfg: UNetDecoderConfig,
                 encoder_channels: dict[str, int]):
        super().__init__()
        self.encoder_channels = encoder_channels
        channels = list(encoder_channels.values())[::-1]
        self._out_channels = channels[-1]
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            skip_channels = channels[i + 1]
            out_channels = channels[i + 1]

            block = UNetDecoderBlock(
                in_channels,
                skip_channels,
                out_channels
            )
            self.blocks.append(block)

    def forward(
            self,
            features: dict[str, torch.Tensor]
    ):
        encoder_features = list(features.values())[::-1]
        x = encoder_features[0]

        for idx, block in enumerate(self.blocks):
            skip = encoder_features[idx + 1]
            x = block(x, skip)
        
        return x

    @property
    def out_channels(self) -> int:
        return self._out_channels