from woname.transforms.base import TransformBase
from typing import List
from .registry import TRANSFORMS

class Compose:
    def __init__(
            self,
            transforms: List[TransformBase]
    ):
        self.transforms = transforms

    def __call__(
            self,
            sample
    ):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    @classmethod
    def from_dicts(cls, transforms: List[dict]) -> "Compose":
        transforms_compose = [
            TRANSFORMS.build(t) 
            for t in transforms
        ]
        return Compose(transforms_compose)