import torch

from woname.architectures.heads.configs import SegmentationHeadConfig
from woname.architectures.heads.registry import HEADS

import woname.architectures.heads.models.segmentation_head


def main():

    cfg = SegmentationHeadConfig(
        num_classes=21
    )

    head = HEADS.build(
        cfg,
        in_channels=64
    )

    x = torch.randn(1, 64, 256, 256)

    y = head(x)

    print(y.shape)


if __name__ == "__main__":

    main()