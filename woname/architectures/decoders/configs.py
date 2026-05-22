from pydantic import BaseModel
from typing import Literal

class UNetDecoderConfig(BaseModel):
    type: Literal["unet_decoder"] = "unet_decoder"
