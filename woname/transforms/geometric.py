import cv2
import random

from woname.transforms.base import TransformBase
from woname.transforms.registry import TRANSFORMS
from woname.transforms.configs import (
    ResizeConfig,
    RandomHorizontalFlipConfig
)

@TRANSFORMS.register("resize", ResizeConfig)
class Resize(TransformBase):
    def __init__(
            self,
            cfg: ResizeConfig
    ):
        super().__init__()
        self.size = cfg.size
    
    def __call__(self, sample):
        sample = sample.copy()
        sample["image"] = cv2.resize(
            sample["image"],
            self.size
        )
        if sample.get("mask") is not None:
            sample["target"] = cv2.resize(
                sample["target"],
                self.size,
                interpolation=cv2.INTER_NEAREST
            )
            sample["mask"] = sample["target"]
        return sample
    

@TRANSFORMS.register("randomhorizontalflip", RandomHorizontalFlipConfig)
class RandomHorizontalFlip(TransformBase):
    def __init__(
            self,
            cfg: RandomHorizontalFlipConfig
    ):
        super().__init__()
        self.p = cfg.p

    def __call__(
            self,
            sample
    ):
        sample = sample.copy()
        if random.random() < self.p:
            sample["image"] = cv2.flip(
                sample["image"], 
                1
            )

            if sample.get("mask") is not None:
                sample["target"] = cv2.flip(
                    sample["target"], 
                    1
                )
                sample["mask"] = sample["target"]
        return sample