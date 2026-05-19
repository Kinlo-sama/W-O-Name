from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, Annotated

# Clases de prueba
class ResNetConfig(BaseModel):
    type: Literal["resnet"] = "resnet"
    depth: int = 50
    pretrained: bool = False


class ConvNeXtConfig(BaseModel):
    type: Literal["convnext"] = "convnext"
    variant: Literal["tiny", "small", "base", "large"]
    pretrained: bool = False


class ViTConfig(BaseModel):
    type: Literal["vit"] = "vit"
    patch_size: int = 16
    embed_dim: int = 768
    pretrained: bool = False

class TinyCNNConfig(BaseModel):
    type: Literal["tiny_cnn"] = "tiny_cnn"

    in_channels: int = 3
    hidden_dim: int = 32
    out_dim: int = 64


BackboneConfig = Annotated[
    Union[
        ResNetConfig, 
        ConvNeXtConfig,
        ViTConfig,
        TinyCNNConfig
    ],
    Field(discriminator="type")
]