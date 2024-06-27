import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(channels=out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


# Non-local block is for capturing long-range dependencies in the image to improve the quality of the generated images. It simply computes the attention between each pixel in the image and updates the pixel value based on the attention weights.
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )  # W_q
        self.k = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )  # W_k
        self.v = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )  # W_v
        self.proj_out = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = self.gn(x)  # GroupNorm
        q = self.q(h_)  # q = W_q * h (query) W_q is a 1x1 convolution
        k = self.k(h_)  # k = W_k * h (key) W_k is a 1x1 convolution
        v = self.v(h_)  # v = W_v * h (value) W_v is a 1x1 convolution

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(input=q, mat2=k)  # q * k^T (T is transpose)
        attn = attn * (int(c) ** (-0.5))  # scaling for better training
        attn = F.softmax(attn, dim=2)  # softmax over the last dimension
        attn = attn.permute(0, 2, 1)  # transpose

        A = torch.bmm(input=v, mat2=attn)  # v * softmax(q * k^T)
        A = A.reshape(b, c, h, w)  # reshape to original shape

        return x + A  # residual connection with attention
