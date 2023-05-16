import torch
import pdb

class Net(torch.nn.Module):
    def __init__(self, depth=4, mult_chan=32, dilate=1, init_conv_kernel_size=7, in_channels=1, out_channels=1):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.dilate = dilate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_conv_kernel_size = init_conv_kernel_size

        self.net_recurse = _Net_recurse(
            n_in_channels=self.in_channels, mult_chan=self.mult_chan,
            depth_parent=self.depth, depth=self.depth, dilate = self.dilate,
            init_conv_kernel_size=self.init_conv_kernel_size
        )
        self.conv_out = torch.nn.Conv3d(
            self.mult_chan, self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        # pdb.set_trace()
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth_parent=0, depth=0,
                 dilate=1, init_conv_kernel_size=7, groups=8):
        """Class for recursive definition of U-network.p

        Parameters
        ----------
        in_channels
            Number of channels for input.
        mult_chan
            Factor to determine number of output channels
        depth
            If 0, this subnet will only be convolutions that double the channel
            count.
        dilate
            Number of blocks in bottleneck with progressive dilation.

        """
        super().__init__()

        self.depth = depth
        self.depth_parent = depth_parent

        if self.depth == depth_parent:
            n_out_channels = mult_chan
            # self.init_conv = torch.nn.Conv3d(
            #     n_in_channels, n_out_channels, kernel_size=(3, 7, 7), padding=init_conv_kernel_size // 2
            # )
            # n_in_channels = n_out_channels
        else:
            n_out_channels = n_in_channels * mult_chan

        if depth == 0:
            self.sub_2conv_more = DilatedBottleneck(n_in_channels, n_out_channels, depth=dilate)
        elif depth > 0:
            groups = 1 if n_out_channels < groups else groups
            self.conv_down = torch.nn.Conv3d(
                n_in_channels, n_out_channels, 2, stride=2
            )
            self.sub_2conv_more = SubNet2Conv(n_out_channels, n_out_channels)
            self.sub_2conv_less = SubNet2Conv(3 * n_out_channels, n_out_channels)
            self.bn0 = torch.nn.GroupNorm(groups, n_out_channels)
            self.relu0 = torch.nn.SiLU(inplace=True)
            self.convt = torch.nn.ConvTranspose3d(
                n_out_channels, n_out_channels, kernel_size=2, stride=2
            )
            self.bn1 = torch.nn.GroupNorm(groups, n_out_channels)
            self.relu1 = torch.nn.SiLU(inplace=True)
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth_parent=depth_parent, depth=(depth - 1))

    def forward(self, x):
        # pdb.set_trace()
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            # if self.depth == self.depth_parent:
            #     x = self.init_conv(x)
            # number of slices must match that in training data or 32??
            x = self.conv_down(x)
            x = self.sub_2conv_more(x)

            x1 = self.sub_u(x)
            # pdb.set_trace()
            x = torch.cat((x, x1), 1)  # concatenate
            x = self.sub_2conv_less(x)
            x = self.convt(x)

        return x


class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out, groups=8):
        super().__init__()
        groups = 1 if n_in < groups else groups

        self.norm1 = torch.nn.GroupNorm(groups, n_in)
        self.activation1 = torch.nn.SiLU(inplace=True)
        self.conv1 = torch.nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(groups, n_out)
        self.activation2 = torch.nn.SiLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.res_conv = torch.nn.Conv3d(n_in, n_out, 1) if n_in != n_out else Identity()

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.activation1(x1)
        x1 = self.conv1(x1)
        x1 = self.norm2(x1)
        x1 = self.activation2(x1)
        x1 = self.conv2(x1)
        return x1 + self.res_conv(x)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Block3d(torch.nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        groups = 2
    ):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(groups, n_in)
        self.activation = torch.nn.SiLU()
        self.project = torch.nn.Conv2d(n_in, n_out, 3, padding = 1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class DilatedBottleneck(torch.nn.Module):
    def __init__(self, n_in, n_out, depth=4, groups=8):
        super().__init__()
        groups = 1 if n_in < groups else groups

        for i in range(depth):
            dilate = 2 ** i
            model = [
                torch.nn.GroupNorm(groups, n_in),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(n_in, n_out, kernel_size=3, padding=dilate, dilation=dilate),
            ]
            self.add_module('bottleneck%d' % (i + 1), torch.nn.Sequential(*model))
            if i == 0:
                n_in = n_out

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output