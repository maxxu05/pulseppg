import torch
from torch import nn
import torch.nn.functional as F


class dilated_conv_net(torch.nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=256,
        bottleneck=32,
        kernel_size=15,
        dilation=1,
        double_receptivefield=5,
        stride=1,
    ):
        super().__init__()
        self.layer0 = dilated_conv_block(
            in_channel, out_channel, bottleneck, kernel_size, dilation, firstlayer=True
        )
        layers_list = []
        # if doubled 5 times, then total rf is 883: 1 + 14 = 15 -> 15 + 14*2 = 43 -> 43 + 14*4 = 99 -> 99 + 14*8 = 211 -> 211 + 14*16 = 435 -> 435 + 14*32 = 883
        # receptive field calculation https://stats.stackexchange.com/questions/265462/whats-the-receptive-field-of-a-stack-of-dilated-convolutions
        for i in range(double_receptivefield):
            if i == double_receptivefield - 1:
                currentlayer_stride = stride
            else:
                currentlayer_stride = 1
            layers_list.append(
                dilated_conv_block(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    bottleneck=bottleneck,
                    kernel_size=kernel_size,
                    dilation=dilation * (2**i),
                    stride=currentlayer_stride,
                )
            )
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x, mask=None):
        if mask == None:
            mask = torch.ones(x.shape, requires_grad=False).to(x.device)
        x0 = self.layer0(x, mask)
        return self.layers(x0)


class dilated_conv_block(torch.nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=64,
        bottleneck=32,
        kernel_size=15,
        dilation=1,
        firstlayer=False,
        stride=1,
    ):
        super().__init__()
        self.activation = nn.GELU()
        self.instnorm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.stride = stride
        if firstlayer:
            self.dilated_conv = PartialConv1d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) // 2 * dilation,
                multi_channel=True,
            )
        else:
            self.bottle = nn.Conv1d(in_channel, bottleneck, kernel_size=1)
            self.dilated_conv = nn.Conv1d(
                bottleneck,
                out_channel,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=(kernel_size - 1) // 2 * dilation,
            )

    def forward(self, x, mask=None):
        if (
            mask == None
        ):  # this implies it is not the first layer, so we can use bottleneck layer and residual connection
            if self.stride != 1:
                return self.instnorm(
                    self.activation(
                        self.dilated_conv(self.bottle(x)) + x[:, :, :: self.stride]
                    )
                )
            else:
                return self.instnorm(
                    self.activation(self.dilated_conv(self.bottle(x)) + x)
                )
        else:
            return self.instnorm(self.activation(self.dilated_conv(x, mask)))


class PartialConv1d(nn.Conv1d):
    def __init__(self, *args, fixratio=True, **kwargs):

        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv1d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels, self.kernel_size[0]
            )
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

        self.fixratio = fixratio

    def forward(self, input, mask_in):
        # import pdb; pdb.set_trace()
        # assert len(input.shape) == 3
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                        ).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2]).to(input)
                else:
                    mask = mask_in.float()

                self.update_mask = F.conv1d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                if self.fixratio:
                    self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
                else:
                    self.mask_ratio = 1

        raw_out = super(PartialConv1d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
