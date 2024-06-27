import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3):
        """
        Initialize the discriminator network.

        Args:
            args (argparse.Namespace): The command-line arguments.
            num_filters_last (int): The number of filters in the last layer of the discriminator.
            n_layers (int): The number of layers in the discriminator.
        """
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(
                args.image_channels,
                num_filters_last,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=4,
                    stride=2 if i < n_layers else 1,
                    padding=1,
                    bias=False,
                ),
            ]
        layers.append(
            nn.Conv2d(
                num_filters_last * num_filters_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
