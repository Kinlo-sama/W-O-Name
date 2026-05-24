from typing import Literal, Optional, Any
from pydantic import BaseModel

from woname.architectures.backbones.configs import BackboneConfig
from woname.architectures.decoders.configs import UNetDecoderConfig
from woname.architectures.heads.configs import SegmentationHeadConfig


class UNetConfig(BaseModel):
    type: Literal["unet"] = "unet"

    backbone: BackboneConfig
    decoder: UNetDecoderConfig
    neck: Optional[Any] = None
    head: SegmentationHeadConfig