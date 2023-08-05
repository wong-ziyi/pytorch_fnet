import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from functools import partial


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Net(nn.Module):
    def __init__(
        self,
        depth=4,
        mult_chan=32,
        dilate=1,
        in_channels=1,
        out_channels=1,
        upsample="convt",
        downsample="stride2_conv",
        activation="relu",
        norm="batch",
        groups=None,
        res_block=False,
        init_embed=None,
        init_kernel_size=3,
    ):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.dilate = dilate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.res_block = res_block

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be one of: 'relu', 'elu', 'selu', 'silu', 'gelu'")

        if norm == "batch":
            self.norm = nn.BatchNorm3d
        elif norm == "instance":
            self.norm = nn.InstanceNorm3d
        elif norm == "layer":
            self.norm = nn.LayerNorm
        elif norm == "group" and groups is not None:
            self.norm = nn.GroupNorm
        else:
            raise ValueError("norm must be 'batch' or 'instance' or 'group'")

        if upsample == "pixel_shuffle":
            self.upsample = partial(PixelShuffleUpsample, scale_factor=2, activation_fn=self.activation)
        elif upsample == "convt":
            self.upsample = partial(nn.ConvTranspose3d, kernel_size=2, stride=2)
        else:
            raise ValueError("upsample must be 'pixel_shuffle' or 'convt'")

        if downsample == "pixel_unshuffle":
            self.downsample = partial(pixel_unshuffle_downsample, scale_factor=2)
        elif downsample == "stride2_conv":
            self.downsample = partial(nn.Conv3d, kernel_size=2, stride=2)
        else:
            raise ValueError("downsample must be 'stride2_conv' or 'pixel_unshuffle'")

        n_in_subnet = self.in_channels
        if init_embed == "conv":
            n_in_subnet = self.mult_chan
            self.init_embed = nn.Conv3d(
                self.in_channels, self.mult_chan, kernel_size=init_kernel_size, padding=init_kernel_size // 2
            )
        elif init_embed == "cross_embed":
            n_in_subnet = self.mult_chan
            self.init_embed = CrossEmbedLayer3D(
                self.in_channels, dim_out=self.mult_chan, kernel_sizes=init_kernel_size, stride=1
            )
        elif init_embed is not None:
            raise ValueError("init_embed must be 'conv' or 'cross_embed'")

        self.net_recurse = _Net_recurse(
            n_in_channels=n_in_subnet,
            mult_chan=self.mult_chan,
            depth_parent=self.depth,
            depth=self.depth,
            dilate=self.dilate,
            upsample=self.upsample,
            downsample=self.downsample,
            activation_fn=self.activation,
            norm_fn=self.norm,
            groups=self.groups,
            res_block=self.res_block,
        )
        self.conv_out = nn.Conv3d(self.mult_chan, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if hasattr(self, "init_embed"):
            x = self.init_embed(x)
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class _Net_recurse(nn.Module):
    def __init__(
        self,
        n_in_channels,
        mult_chan=2,
        depth_parent=0,
        depth=0,
        dilate=1,
        upsample=None,
        downsample=None,
        activation_fn=None,
        norm_fn=None,
        groups=None,
        res_block=False,
    ):
        """Class for recursive definition of U-network.

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
        self.dilate = dilate
        self.res_block = res_block
        self.norm = norm_fn or nn.BatchNorm3d
        self.activation = activation_fn or nn.ReLU(inplace=True)
        self.upsample = upsample or partial(nn.ConvTranspose3d, kernel_size=2, stride=2)
        self.downsample = downsample or partial(nn.Conv3d, kernel_size=2, stride=2)

        n_out_channels = mult_chan if self.depth == depth_parent else n_in_channels * mult_chan

        if depth == 0:
            self.sub_2conv_more = DilatedBottleneck(
                n_in_channels,
                n_out_channels,
                depth=self.dilate,
                activation_fn=self.activation,
                norm_fn=self.norm,
                groups=groups,
            )

        elif depth > 0:

            if groups is not None:
                groups = 1 if n_out_channels < groups else groups
                self.bn0, self.bn1 = self.norm(groups, n_out_channels), self.norm(groups, n_out_channels)
            else:
                self.bn0, self.bn1 = self.norm(n_out_channels), self.norm(n_out_channels)

            self.relu0 = self.activation
            self.relu1 = self.activation

            self.conv_down = self.downsample(n_out_channels, n_out_channels)
            self.convt = self.upsample(2 * n_out_channels, n_out_channels)

            self.sub_2conv_more = SubNet2Conv(
                n_in_channels,
                n_out_channels,
                activation_fn=self.activation,
                norm_fn=self.norm,
                groups=groups,
                res_block=res_block,
            )
            self.sub_2conv_less = SubNet2Conv(
                2 * n_out_channels,
                n_out_channels,
                activation_fn=self.activation,
                norm_fn=self.norm,
                groups=groups,
                res_block=res_block,
            )

            self.sub_u = _Net_recurse(
                n_out_channels,
                mult_chan=2,
                depth_parent=depth_parent,
                depth=(depth - 1),
                dilate=self.dilate,
                upsample=self.upsample,
                downsample=self.downsample,
                activation_fn=self.activation,
                norm_fn=self.norm,
                groups=groups,
                res_block=self.res_block,
            )

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


class SubNet2Conv(nn.Module):
    def __init__(self, n_in, n_out, activation_fn=None, norm_fn=None, groups=None, res_block=False):
        super().__init__()
        self.res_block = res_block
        norm_fn = norm_fn or nn.BatchNorm3d
        activation_fn = activation_fn or nn.ReLU(inplace=True)

        if groups is not None:
            groups = 1 if n_in < groups else groups
            self.bn1 = norm_fn(groups, n_out)
            self.bn2 = norm_fn(groups, n_out)
        else:
            self.bn1 = norm_fn(n_out)
            self.bn2 = norm_fn(n_out)

        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.relu1 = activation_fn
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.relu2 = activation_fn

        if res_block:
            self.res_conv = nn.Conv3d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
            if res_block == "regress":
                self.beta = nn.Parameter(torch.zeros((1, n_out, 1, 1, 1)), requires_grad=True)
                self.gamma = nn.Parameter(torch.zeros((1, n_out, 1, 1, 1)), requires_grad=True)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        # learned skip connection over first convolution
        if self.res_block == "regress":
            y = self.res_conv(x) + h * self.beta
        # skip connection over first convolution
        elif self.res_block == "x2":
            y = self.res_conv(x) + h
        # covers both no skip connection and classic skip connection
        else:
            y = h

        h = self.conv2(y)
        h = self.bn2(h)
        h = self.relu2(h)

        if not self.res_block:
            y = h
        # learned skip connection over second convolution
        elif self.res_block == "regress":
            y = y + h * self.gamma
        # skip connection over second convolution
        elif self.res_block == "x2":
            y = y + h
        # classic skip connection over both convolutions
        else:
            y = self.res_conv(x) + h

        return y


class DilatedBottleneck(nn.Module):
    def __init__(self, n_in, n_out, depth=4, activation_fn=None, norm_fn=None, groups=None):
        super().__init__()
        norm_fn = norm_fn or nn.BatchNorm3d
        activation_fn = activation_fn or nn.ReLU(inplace=True)

        if groups is not None:
            groups = 1 if n_in < groups else groups
            norm_fn = norm_fn(groups, n_out)
        else:
            norm_fn = norm_fn(n_out)

        for i in range(depth):
            dilate = 2**i
            model = [
                nn.Conv3d(n_in, n_out, kernel_size=3, padding=dilate, dilation=dilate),
                norm_fn,
                activation_fn,
            ]
            self.add_module(f"bottleneck{i + 1}", nn.Sequential(*model))
            if i == 0:
                n_in = n_out

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class PixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        scale_factor=2,
        activation_fn=None,
    ):
        super().__init__()
        self.scale_squared = scale_factor**3
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * self.scale_squared, 1)
        activation_fn = activation_fn or nn.ReLU(inplace=True)

        self.net = nn.Sequential(
            conv,
            activation_fn,
            Rearrange("b (c r s p) f h w -> b c (f p) (h r) (w s)", p=scale_factor, r=scale_factor, s=scale_factor),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, *rest_dims = conv.weight.shape
        conv_weight = torch.empty(o // self.scale_squared, i, *rest_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o r) ...", r=self.scale_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = self.net(x)
        return x


def pixel_unshuffle_downsample(dim, dim_out, scale_factor=2):
    return nn.Sequential(
        Rearrange(
            "b c (f s1) (h s2) (w s3) -> b (c s1 s2 s3) f h w", s1=scale_factor, s2=scale_factor, s3=scale_factor
        ),
        nn.Conv3d(dim * scale_factor**3, dim_out, 1),
    )


class CrossEmbedLayer3D(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv3d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)
