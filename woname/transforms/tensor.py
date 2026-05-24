import torch
from woname.transforms.base import TransformBase
from woname.transforms.configs import ToTensorConfig
from woname.transforms.registry import TRANSFORMS

@TRANSFORMS.register("totensor", ToTensorConfig)
class ToTensor(TransformBase):
    def __init__(
            self,
            cfg: ToTensorConfig
        ):
        super().__init__()
    
    def __call__(self, sample):
        sample["image"] = torch.from_numpy(
            sample["image"]
        ).permute(2,0,1).float()

        if sample.get("mask") is not None:
            sample["mask"] = torch.from_numpy(
                sample["mask"]
            ).unsqueeze(0).float()
            sample["target"] = sample["mask"]

        return sample  