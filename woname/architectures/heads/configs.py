from pydantic import BaseModel
from typing import Literal

class SegmentationHeadConfig(BaseModel):
    type: Literal["segmentation_head"] = "segmentation_head"
    num_classes: int 