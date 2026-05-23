from woname.transforms.base import TransformBase

class Normalize(TransformBase):
    def __init__(
            self,
            mean,
            std
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = sample.copy()

        sample["image"] = (
            sample["image"] - self.mean 
        ) / self.std

        return sample