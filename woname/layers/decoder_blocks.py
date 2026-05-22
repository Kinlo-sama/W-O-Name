import torch
import torch.nn as nn
from woname.layers.conv_blocks import DoubleConv

class UNetDecoderBlock(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 skip_channels:int,
                 out_channels:int
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d( # <- Duplicar resolución espacial
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, 
                x:torch.Tensor, 
                skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
