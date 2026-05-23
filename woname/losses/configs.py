from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union

class DiceLossConfig(BaseModel):
    type: Literal["dice_loss"] = "dice_loss"
    smooth: float = 1e-6

class BCELossConfig(BaseModel):
    type: Literal["bce_loss"] = "bce_loss"

class DiceBCELossConfig(BaseModel):
    type: Literal["dice_bce_loss"] = "dice_bce_loss"
    dice_weight: float = 1.0
    bce_weight: float = 1.0

LossConfig = Annotated[
    Union[
        DiceLossConfig,
        BCELossConfig,
        DiceBCELossConfig
    ],
    Field(discriminator="type")
]