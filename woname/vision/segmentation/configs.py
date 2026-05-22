from typing import Literal
from pydantic import BaseModel

from woname.architectures.backbones.configs import BackboneConfig
from woname.architectures.decoders.configs import UNetDecoderConfig
from woname.architectures.heads.configs import SegmentationHeadConfig


class UNetConfig(BaseModel):
    type: Literal["unet"] = "unet"

    backbone: BackboneConfig
    decoder: UNetDecoderConfig
    head: SegmentationHeadConfig