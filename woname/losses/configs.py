from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union

class DiceLossConfig(BaseModel):
    type: Literal["dice_loss"] = "dice_loss"
    smooth: float = Field(default=1e-6, gt=0)

class BCELossConfig(BaseModel):
    type: Literal["bce_loss"] = "bce_loss"

class DiceBCELossConfig(BaseModel):
    type: Literal["dice_bce_loss"] = "dice_bce_loss"
    dice_weight: float = Field(default=1.0, gt=0)
    bce_weight: float = Field(default=1.0, gt=0)
    dice_cfg: DiceLossConfig = Field(default_factory=DiceLossConfig)
    bce_cfg: BCELossConfig = Field(default_factory=BCELossConfig)
 

LossConfig = Annotated[
    Union[
        DiceLossConfig,
        BCELossConfig,
        DiceBCELossConfig
    ],
    Field(discriminator="type")
]