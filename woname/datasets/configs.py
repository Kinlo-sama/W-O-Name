from pydantic import BaseModel, Field
from typing import Union, Annotated, Optional, Literal, Any

class SegmentationDatasetConfig(BaseModel):
    type: Literal["segmentation_dataset"] = "segmentation_dataset"
    images_dir: str
    masks_dir: str

    img_suffix: str = ".png"
    mask_suffix: str = ".png"

    images_size: Optional[
        tuple[int, int]
    ] = None
    transforms: Optional[Any]= None

DatasetConfig = Annotated[
    Union[
        SegmentationDatasetConfig,
    ],
    Field(discriminator="type")
]