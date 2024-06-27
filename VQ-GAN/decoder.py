import torch.nn as nn
from helper import (
    GroupNorm,
    Swish,
    ResidualBlock,
    UpSampleBlock,
    NonLocalBlock,
)


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != len(channels) - 1:
                layers.append(
                    UpSampleBlock(out_channels)
                )  # Upsample block for all but the last stage
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(
            nn.Conv2d(
                in_channels, args.image_channels, kernel_size=3, stride=1, padding=1
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        self.model(x)
