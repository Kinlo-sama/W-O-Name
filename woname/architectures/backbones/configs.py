from pydantic import BaseModel, Field, RootModel
from typing import Literal, Optional, Union, Annotated

class BackboneConfigBase(BaseModel):
    weights: Optional[str] = None

# Backbones tests
class ResNetConfig(BackboneConfigBase):
    type: Literal["resnet"] = "resnet"
    depth: int = 50
    pretrained: bool = False


class ConvNeXtConfig(BackboneConfigBase):
    type: Literal["convnext"] = "convnext"
    variant: Literal["tiny", "small", "base", "large"]
    pretrained: bool = False


class ViTConfig(BackboneConfigBase):
    type: Literal["vit"] = "vit"
    patch_size: int = 16
    embed_dim: int = 768
    pretrained: bool = False

class TinyCNNConfig(BackboneConfigBase):
    type: Literal["tiny_cnn"] = "tiny_cnn"

    in_channels: int = 3
    hidden_dim: int = 32
    out_dim: int = 64

class UNetEncoderConfig(BackboneConfigBase):
    type: Literal["unet_encoder"] = "unet_encoder"

    in_channels: int = 3
    base_channels: int = 64
    num_stages: int = 4

BackboneConfig = Annotated[
    Union[
        ResNetConfig, 
        ConvNeXtConfig,
        ViTConfig,
        TinyCNNConfig,
        UNetEncoderConfig,
    ],
    Field(discriminator="type")
]