import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from woname.architectures.backbones.configs import (
    UNetEncoderConfig
)

from woname.architectures.decoders.configs import (
    UNetDecoderConfig
)

from woname.architectures.heads.configs import (
    SegmentationHeadConfig
)

from woname.vision.segmentation.configs import (
    UNetConfig
)

from woname.vision.segmentation.registry import (
    SEGMENTATION_MODELS
)

from woname.losses.registry import LOSSES

from woname.core.engine.trainer import Trainer

from woname.core.engine.configs import (
    TrainerConfig
)
from woname.evaluators.configs import DiceScoreConfig, IoUConfig, PixelAccuracyConfig



class DummySegmentationDataset(Dataset):

    def __init__(
        self,
        length: int = 20
    ):

        self.length = length

    def __len__(self):

        return self.length

    def __getitem__(
        self,
        idx
    ):
        
        image = torch.randn(
            3,
            128,
            128
        )

        mask = torch.randint(
            0,
            2,
            (1, 128, 128)
        ).float()

        sample = dict(
            image=image,
            mask=mask,
            target=mask
        )
        return sample
    
dataset = DummySegmentationDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model_cfg = UNetConfig(

    backbone=UNetEncoderConfig(),

    decoder=UNetDecoderConfig(),

    head=SegmentationHeadConfig(
        num_classes=1
    )
)

model = SEGMENTATION_MODELS.build(
    model_cfg
)

from woname.losses.configs import (
    DiceBCELossConfig
)

loss_cfg = DiceBCELossConfig()

criterion = LOSSES.build(
    loss_cfg
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

trainer_cfg = TrainerConfig(
    epochs=3,
    device="cpu",
    evaluators=[
        IoUConfig(),
        PixelAccuracyConfig(),
        DiceScoreConfig()
    ]
)

trainer = Trainer(
    trainer_cfg
)

trainer.fit(
    model=model,
    train_loader=loader,
    optimizer=optimizer,
    criterion=criterion
)