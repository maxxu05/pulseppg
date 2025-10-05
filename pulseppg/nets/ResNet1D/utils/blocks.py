import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding="same",
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding="same",
        )

        # Force projection if N input channels != N channels OR if strides are not all 1's
        if in_channels != out_channels or self.stride != 1:
            self.projection_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=self.stride,
            )
        else:
            self.projection_conv = torch.nn.Identity()

    def forward(self, x):

        # the first conv
        out = x
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)

        # the second conv
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # residual con
        out = out + self.projection_conv(x)

        return out
