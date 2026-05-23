from typing import Literal
from pydantic import BaseModel, Field
from typing import Union, Annotated

class DiceScoreConfig(BaseModel):
    type: Literal["dice_score"] = "dice_score"
    threshold: float = 0.5
    smooth: float = 1e-6

class IoUConfig(BaseModel):
    type: Literal["iou"] = "iou"
    threshold: float = 0.5
    smooth: float = 1e-6

class PixelAccuracyConfig(BaseModel):
    type: Literal["pixel_accuracy"] = "pixel_accuracy"
    threshold: float = 0.5

MetricConfig = Annotated[
    Union[
        DiceScoreConfig,
        IoUConfig,
        PixelAccuracyConfig
    ],
    Field(discriminator="type")
]