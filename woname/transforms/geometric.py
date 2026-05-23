import cv2
from woname.transforms.base import TransformBase
import random

class Resize(TransformBase):
    def __init__(
            self,
            size
    ):
        super().__init__()
        self.size = size
    
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
    

class RandomHorizontalFlip(TransformBase):
    def __init__(
            self,
            p: float = 0.5
    ):
        super().__init__()
        self.p = p

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