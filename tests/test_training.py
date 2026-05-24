import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from woname.vision.segmentation.registry import SEGMENTATION_MODELS
from woname.losses.registry import LOSSES
from woname.transforms.compose import Compose

from woname.core.engine.trainer import Trainer
from woname.core.engine.configs import TrainerConfig


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

compose = Compose.from_dicts([
    {"type": "resize", "size": (512,512)},
    {"type": "randomhorizontalflip"}
])

backbone = {
    "type":"unet_encoder",
}

decoder = {
    "type":"unet_decoder",
}

head = {
    "type":"segmentation_head",
    "num_classes":1
}
model = SEGMENTATION_MODELS.build({
    "type":"unet",
    "backbone": backbone,
    "decoder": decoder,
    "head": head
})

criterion = LOSSES.build({
    "type":"dice_bce_loss"
})

evaluators = [
    {"type":"iou"},
    {"type": "pixel_accuracy"},
    {"type": "dice_score"}
]

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

trainer_cfg = TrainerConfig(
    epochs=3,
    device="cpu",
    evaluators=evaluators
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