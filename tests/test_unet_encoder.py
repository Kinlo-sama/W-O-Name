import torch

from woname.architectures.backbones.configs import UNetEncoderConfig

from woname.architectures.backbones.registry import BACKBONES

# Importante:
# fuerza registro del modelo
import woname.architectures.backbones.models.unet_encoder


def main():

    cfg = UNetEncoderConfig(
        in_channels=3,
        base_channels=64,
        num_stages=4
    )

    model = BACKBONES.build(cfg)

    print(model)

    x = torch.randn(1, 3, 256, 256)

    outputs = model(x)

    print("\n=== OUTPUT FEATURES ===\n")

    for name, feat in outputs.items():

        print(f"{name}: {feat.shape}")

    print("\n=== OUT CHANNELS ===\n")

    print(model.out_channels)


if __name__ == "__main__":

    main()