from pydantic import BaseModel, field_validator, Field
from typing import Union, Annotated, Optional, Literal, Any

class ResizeConfig(BaseModel):
    type: Literal["resize"] = "resize"
    size: tuple[int, int]

    @field_validator("size")
    @classmethod
    def validated_size(cls, v):
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(
                f"Size values must be greater than 0, got {v}"
            )
        return v

class RandomHorizontalFlipConfig(BaseModel):
    type: Literal["randomhorizontalflip"] = "randomhorizontalflip"
    p: float = Field(ge=0, default=0.5, le=1)

class ToTensorConfig(BaseModel):
    type: Literal["totensor"] = "totensor"

Transforms = Annotated[
    Union[
        ResizeConfig,
        RandomHorizontalFlipConfig
    ],
    Field(discriminator="type")
]