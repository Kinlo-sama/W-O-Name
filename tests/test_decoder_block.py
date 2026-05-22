import torch

from woname.layers.decoder_blocks import UNetDecoderBlock


def main():

    block = UNetDecoderBlock(
        in_channels=512,
        skip_channels=256,
        out_channels=256
    )

    x = torch.randn(1, 512, 32, 32)

    skip = torch.randn(1, 256, 64, 64)

    y = block(x, skip)

    print(y.shape)


if __name__ == "__main__":

    main()