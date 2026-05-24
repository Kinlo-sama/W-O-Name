from pydantic import BaseModel
from typing import Literal, Dict

class UNetDecoderConfig(BaseModel):
    type: Literal["unet_decoder"] = "unet_decoder"