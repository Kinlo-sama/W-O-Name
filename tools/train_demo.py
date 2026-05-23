import torch
from torch.utils.data import DataLoader

from woname.architectures.backbones.configs import UNetEncoderConfig
from woname.architectures.decoders.configs import UNetDecoderConfig
from woname.architectures.heads.configs import SegmentationHeadConfig
from woname.vision.segmentation.configs import UNetConfig

from woname.vision.segmentation.registry import SEGMENTATION_MODELS
from woname.losses.registry import LOSSES

from woname.core.engine.trainer import Trainer
from woname.core.engine.configs import TrainerConfig

from woname.datasets.segmentation.segmentation_dataset import SegmentationClass
from woname.datasets.configs import SegmentationDatasetConfig

model_cfg = UNetConfig(
    backbone=UNetEncoderConfig(),
    decoder=UNetDecoderConfig(),
    head=SegmentationHeadConfig(num_classes=1)
)

model = SEGMENTATION_MODELS.build(model_cfg)
from woname.transforms.geometric import Resize, RandomHorizontalFlip
from woname.transforms.photometric import Normalize
from woname.transforms.tensor import ToTensor
from woname.transforms.compose import Compose
from woname.losses.configs import DiceBCELossConfig

from woname.evaluators.configs import IoUConfig, DiceScoreConfig, PixelAccuracyConfig


transforms = Compose([

    Resize((256, 256)),
    RandomHorizontalFlip(),
    ToTensor()
])

criterion = LOSSES.build(DiceBCELossConfig())

dataset = SegmentationClass(
    SegmentationDatasetConfig(
        images_dir="data/SegmentationRoads/images",
        masks_dir="data/SegmentationRoads/masks",
        img_suffix=".jpg",
        mask_suffix=".png",
        transforms=transforms
    )
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

trainer = Trainer(
    TrainerConfig(
        epochs=3,
        device="cpu",
        evaluators=[
            IoUConfig(),
            PixelAccuracyConfig(),
            DiceScoreConfig()
        ]
    )
)

trainer.fit(
    model=model,
    train_loader=loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=criterion
)