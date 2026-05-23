import torch

from woname.evaluators.segmentation.dice_score import DiceScore
from woname.evaluators.segmentation.iou import IoU
from woname.evaluators.segmentation.pixel_accuracy import (
    PixelAccuracy
)


def main():

    logits = torch.randn(2, 1, 256, 256)

    targets = torch.randint(
        0,
        2,
        (2, 1, 256, 256)
    ).float()

    dice = DiceScore()

    iou = IoU()

    acc = PixelAccuracy()

    print("Dice:", dice(logits, targets))

    print("IoU:", iou(logits, targets))

    print("Accuracy:", acc(logits, targets))


if __name__ == "__main__":

    main()