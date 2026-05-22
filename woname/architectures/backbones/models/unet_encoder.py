import torch
import torch.nn  as nn

from woname.layers.conv_blocks import DoubleConv

from ..base import BackboneBase
from ..registry import BACKBONES
from ..configs import UNetEncoderConfig

@BACKBONES.register("unet_encoder")
class UNetEncoder(BackboneBase):
    def __init__(self, cfg: UNetEncoderConfig):
        super().__init__()
        self.cfg = cfg

        channels = [
            cfg.base_channels * (2 ** i)
            for i in range(cfg.num_stages)
        ]
 
        self._out_channels = {
            f"stage{i+1}":ch 
            for i, ch in enumerate(channels)
        }
        self._num_stages = cfg.num_stages
        self.stages = nn.ModuleList()
        
        in_channels = cfg.in_channels
        for out_channels in channels:
            stage = DoubleConv(
                in_channels,
                out_channels
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        
        features = {}
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            features[f"stage{idx+1}"] = x

            if idx < len(self.stages) - 1:
                x = self.pool(x)
        return features
    
    @property
    def out_channels(self):
        return self._out_channels