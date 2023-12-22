import torch


class Net(torch.nn.Module):
    def __init__(self, depth=4, mult_chan=32, dilate=1, in_channels=1, out_channels=1):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.dilate = dilate
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net_recurse = _Net_recurse(
            n_in_channels=self.in_channels, mult_chan=self.mult_chan, depth_parent=self.depth, depth=self.depth, dilate = self.dilate
        )
        self.conv_out = torch.nn.Conv3d(
            self.mult_chan, self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth_parent=0, depth=0, dilate=1):
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

        if self.depth == depth_parent:
            n_out_channels = mult_chan
        else:
            n_out_channels = n_in_channels * mult_chan

        if depth == 0:
            self.sub_2conv_more = DilatedBottleneck(n_in_channels, n_out_channels, depth=dilate)
        elif depth > 0:
            self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
            self.sub_2conv_less = SubNet2Conv(2 * n_out_channels, n_out_channels)
            self.conv_down = torch.nn.Conv3d(
                n_out_channels, n_out_channels, 2, stride=2
            )
            self.bn0 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu0 = torch.nn.ReLU(inplace=True)
            self.convt = torch.nn.ConvTranspose3d(
                2 * n_out_channels, n_out_channels, kernel_size=2, stride=2
            )
            self.bn1 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth_parent=depth_parent, depth=(depth - 1))

    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            # number of slices must match that in training data or 32??
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less


class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU(inplace=True)
        #self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #x = self.dropout(x)
        return x

    
class DilatedBottleneck(torch.nn.Module):
    def __init__(self, n_in, n_out, depth=4):
        super().__init__()

        for i in range(depth):
            dilate = 2 ** i
            model = [
                torch.nn.Conv3d(n_in, n_out, kernel_size=3, padding=dilate, dilation=dilate),
                torch.nn.BatchNorm3d(n_out),
                torch.nn.ReLU(inplace=True)
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