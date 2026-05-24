from pathlib import Path
import cv2
import torch

from woname.datasets.base import DatasetBase
from woname.datasets.registry import DATASETS
from woname.datasets.configs import SegmentationDatasetConfig

@DATASETS.register("segmentation_dataset", SegmentationDatasetConfig)
class SegmentationClass(DatasetBase):
    def __init__(
            self,
            cfg: SegmentationDatasetConfig
    ):
        self.cfg = cfg
        self.transforms = cfg.transforms
        self.imgs_dir = Path(
            cfg.images_dir
        )
        self.masks_dir = Path(
            cfg.masks_dir
        )
        self.image_paths =   sorted(
            list(
                self.imgs_dir.glob(
                    f"*{cfg.img_suffix}"
                )
            )
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = (
            self.masks_dir /
            f"{image_path.stem}{self.cfg.mask_suffix}"
        )

        image = cv2.imread(
            str(image_path)
        )
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )
        sample = {
            "image":image,
            "mask": mask,
            "target": mask,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample