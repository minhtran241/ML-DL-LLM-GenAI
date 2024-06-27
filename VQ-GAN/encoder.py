import torch.nn as nn
from helper import (
    GroupNorm,
    Swish,
    ResidualBlock,
    DownSampleBlock,
    NonLocalBlock,
)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]  # Channels for each stage
        attn_resolutions = [16]  # Attention resolutions for non-local block
        num_ress_blocks = 2  # Number of residual blocks in each stage
        resolution = 256  # Input resolution
        layers = [
            nn.Conv2d(
                args.image_channels, channels[0], kernel_size=3, stride=1, padding=1
            )
        ]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_ress_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if (
                i != len(channels) - 2
            ):  # Add downsample block for all but the last stage because we ne
                layers.append(DownSampleBlock(channels[i + 1]))
                resolution = resolution // 2

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(
            nn.Conv2d(channels[-1], args.latent_dim, kernel_size=3, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
