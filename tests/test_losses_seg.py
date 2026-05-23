import torch

from woname.losses.segmentation.dice import DiceLoss
from woname.losses.segmentation.bce import BCELoss
from woname.losses.segmentation.dice_bce import DiceBCELoss


def main():

    logits = torch.randn(2, 1, 256, 256)

    targets = torch.randint(
        0,
        2,
        (2, 1, 256, 256)
    ).float()

    dice = DiceLoss()

    bce = BCELoss()

    combo = DiceBCELoss()

    print("Dice:", dice(logits, targets))

    print("BCE:", bce(logits, targets))

    print("Dice+BCE:", combo(logits, targets))


if __name__ == "__main__":

    main()