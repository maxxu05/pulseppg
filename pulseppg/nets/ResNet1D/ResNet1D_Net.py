"""
ResNet1D for time-series via pytorch
 
Maxwell A. Xu
"""
import torch
import torch.nn as nn

from pulseppg.nets.ResNet1D.utils.blocks import ResBlock


class Net(nn.Module):
    """
    ResNet1D for a time-series embedding using PyTorch.

    This class implements a 1D ResNet for processing time-series data. The architecture consists of an
    initial convolutional block, followed by a series of residual blocks, and a final pooling layer
    to produce the output.

    Args:
        finalpool (str): Type of pooling to use at the end of the network.
            Options:
                - "avg" for average pooling
                - "max" for max pooling
        init_conv (dict): Configuration for the initial convolutional layer.
            Keys:
                - in_channels (int): Number of input channels.
                - base_filters (int): Number of filters in the initial convolutional layer.
                - kernel_size (int): Size of the convolutional kernel.
                - stride (int): Stride of the convolutional kernel.
        init_maxpool (dict): Configuration for the initial max pooling layer.
            Keys:
                - kernel_size (int): Size of the pooling window.
                - stride (int): Stride of the pooling window.
        blocks (list of dict): Configuration for the residual blocks.
            Each dictionary contains:
                - n (int): Number of blocks.
                - in_channels (int): Number of input channels for the block.
                - base_filters (int): Number of filters in each convolutional layer.
                - kernel_size (int): Size of the convolutional kernel.
                - stride (int): Stride of the convolutional kernel.

    Input Shape:
        x (torch.Tensor): Input tensor of shape `(n_samples, n_channels, n_length)`, where:
            - n_samples: Number of samples in the batch.
            - n_channels: Number of input channels.
            - n_length: Length of the time-series data.

    Output Shape:
        torch.Tensor: Output tensor of shape `(n_samples, final_channels)`, representing final embedding

    """

    def __init__(self, finalpool, init_conv, init_maxpool, blocks):
        super().__init__()

        # first block
        self.first_block_conv = nn.Conv1d(
            in_channels=init_conv["in_channels"],
            out_channels=init_conv["base_filters"],
            kernel_size=init_conv["kernel_size"],
            stride=init_conv["stride"],
            padding=init_conv["padding"],
        )
        self.first_block_bn = nn.BatchNorm1d(init_conv["base_filters"])
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = nn.MaxPool1d(
            kernel_size=init_maxpool["kernel_size"], stride=init_maxpool["stride"]
        )

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for block in blocks:
            for i_block in range(block["n"]):
                if i_block == 0:
                    in_channels = block["in_channels"]
                else:
                    in_channels = block["base_filters"]

                tmp_block = ResBlock(
                    in_channels=in_channels,
                    out_channels=block["base_filters"],
                    kernel_size=block["kernel_size"],
                    stride=block["stride"],
                )

                self.basicblock_list.append(tmp_block)

        self.finalpool = finalpool

    def forward(self, x):

        out = x

        out = self.first_block_conv(out)
        out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)

        # residual blocks, every block has two conv
        for block in self.basicblock_list:
            out = block(out)

        if self.finalpool == "avg":
            out = torch.mean(out, dim=-1)
        elif self.finalpool == "max":
            out = torch.max(out, dim=-1)[0]
        else:
            print("invalid pool")
            import sys

            sys.exit()

        return out
