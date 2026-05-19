import torch

from woname.architectures.backbones.configs import TinyCNNConfig
from woname.architectures.backbones.registry import build_backbone

# Import models so registry gets populated
from woname.architectures.backbones.models import *


cfg = TinyCNNConfig()

model = build_backbone(cfg)

x = torch.randn(1, 3, 224, 224)

outputs = model(x)

print(outputs.keys())

for name, feat in outputs.items():
    print(name, feat.shape)