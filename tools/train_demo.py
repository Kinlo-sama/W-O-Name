import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from woname.vision.segmentation.registry import SEGMENTATION_MODELS
from woname.losses.registry import LOSSES
from woname.datasets.registry import DATASETS

from woname.transforms.compose import Compose
from woname.core.engine.trainer import Trainer
from woname.core.engine.configs import TrainerConfig
from woname.datasets.segmentation.segmentation_dataset import SegmentationClass
from woname.datasets.configs import SegmentationDatasetConfig


transforms = Compose.from_dicts([
    {"type": "resize", "size": (512,512)},
    {"type": "randomhorizontalflip"},
    {"type": "totensor"}
])

dataset_cfg = {
    "type": "segmentation_dataset",
    "images_dir": "data/SegmentationRoads/images",
    "masks_dir": "data/SegmentationRoads/masks",
    "img_suffix": ".jpg",
    "mask_suffix": ".png",
    "images_size": (512,512),
    "transforms": transforms
}

dataset = DATASETS.build(dataset_cfg)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

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