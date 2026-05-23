import numpy as np
from woname.transforms.geometric import Resize, RandomHorizontalFlip
from woname.transforms.photometric import Normalize
from woname.transforms.tensor import ToTensor
from woname.transforms.compose import Compose

sample = {

    "image": np.random.randint(
        0,
        255,
        (128, 128, 3),
        dtype=np.uint8
    ),

    "mask": np.random.randint(
        0,
        2,
        (128, 128),
        dtype=np.uint8
    )
}


transforms = Compose([

    Resize((256, 256)),
    RandomHorizontalFlip(),
    ToTensor()
])

output = transforms(sample)

print(
    output["image"].shape
)

print(
    output["mask"].shape
)