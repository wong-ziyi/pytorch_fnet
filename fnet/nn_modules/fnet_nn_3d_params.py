import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from functools import partial


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def act_inplace(act):
    if act == nn.GELU or act == nn.Identity:
        return act()
    else:
        return act(inplace=True)


def get_act(act_str):
    if act_str == "relu":
        return nn.ReLU
    elif act_str == "elu":
        return nn.ELU
    elif act_str == "selu":
        return nn.SELU
    elif act_str == "silu":
        return nn.SiLU
    elif act_str == "gelu":
        return nn.GELU
    elif act_str == "null":
        return None
    else:
        raise ValueError("activation must be one of: 'relu', 'elu', 'selu', 'silu', 'gelu'")


def get_norm(norm_str):
    if norm_str == "batch":
        return nn.BatchNorm3d
    elif norm_str == "instance":
        return nn.InstanceNorm3d
    elif norm_str == "layer":
        return nn.LayerNorm
    elif norm_str == "group":
        return nn.GroupNorm
    elif norm_str == "null":
        return None
    elif norm_str == "identity":
        return nn.Identity
    else:
        raise ValueError("norm must be 'batch' or 'instance' or 'group'")


def get_ca(chan_attn, n_ch):
    if chan_attn == "ca":
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=n_ch, out_channels=n_ch // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=n_ch // 2, out_channels=n_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Sigmoid()
        )
    else:
        return nn.Identity()


def repeat_block(n_repeat, block, n_in, n_out):
    repeated_block = [block(n_in, n_out)]
    for _ in range(n_repeat - 1):
        repeated_block.append(block(n_out, n_out))
    return nn.Sequential(*repeated_block)


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
        double_down=False,
        activation="relu",
        norm="batch",
        order="cna",
        chan_attn=None,
        groups=None,
        block=None,
        mid_block=None,
        res_block=False,
        init_embed=None,
        init_kernel_size=3,
        block_kwargs=None,
    ):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.dilate = dilate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_down = double_down
        self.groups = groups
        self.res_block = res_block
        self.order = order

        self.activation = get_act(activation)
        self.norm = get_norm(norm)

        if chan_attn == "ca" or (res_block and "ca" in res_block):
            self.chan_attn = partial(get_ca, "ca")
        else:
            self.chan_attn = None

        if block == "plain":
            self.block = partial(PlainBaselineBlock, **default(block_kwargs, {}))
            self.mid_block = partial(PlainBaselineBlock, **default(block_kwargs, {}))
        else:
            self.block = partial(SubNet2Conv, activation_fn=self.activation, norm_fn=self.norm, order=self.order, chan_attn=self.chan_attn, groups=self.groups, res_block=self.res_block)
            if mid_block == "dilated":
                self.mid_block = partial(DilatedBottleneck, activation_fn=self.activation, norm_fn=self.norm, order=self.order, chan_attn=self.chan_attn, groups=self.groups, depth=self.dilate)
            elif len(mid_block) == 2 and mid_block[0] == "x":
                single_block = partial(SubNet2Conv, activation_fn=self.activation, norm_fn=self.norm, order=self.order, chan_attn=self.chan_attn, groups=self.groups, res_block=self.res_block)
                self.mid_block = partial(repeat_block, int(mid_block[1]), single_block)
            else:
                self.mid_block = partial(SubNet2Conv, activation_fn=self.activation, norm_fn=self.norm, order=self.order, chan_attn=self.chan_attn, groups=self.groups, res_block=self.res_block)

        if upsample == "pixel_shuffle":
            self.upsample = partial(PixelShuffleUpsample, scale_factor=2, activation_fn=self.activation)
        elif upsample == "pixel_shuffle_noact":
            self.upsample = partial(PixelShuffleUpsample, scale_factor=2)
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
            init_kernel_size = init_kernel_size or 3
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
            upsample=self.upsample,
            downsample=self.downsample,
            double_down=self.double_down,
            activation_fn=self.activation,
            norm_fn=self.norm,
            groups=self.groups,
            block=self.block,
            mid_block=self.mid_block,
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
        upsample=None,
        downsample=None,
        double_down=False,
        activation_fn=None,
        norm_fn=None,
        groups=None,
        block=None,
        mid_block=None,
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
        self.block = block
        self.mid_block = mid_block
        self.norm = norm_fn or nn.Identity
        self.activation = activation_fn or nn.Identity
        self.double_down = double_down
        self.downsample = downsample or partial(nn.Conv3d, kernel_size=2, stride=2)
        self.upsample = upsample or partial(nn.ConvTranspose3d, kernel_size=2, stride=2)

        if self.depth != depth_parent:
            n_out_channels = n_in_channels * mult_chan
        else: 
            # only first level
            n_out_channels = mult_chan

        if depth == 0:
            n_in_conv = n_in_channels
            n_out_conv = n_out_channels if not self.double_down else n_in_channels
            self.sub_2conv_more = self.mid_block(
                n_in_conv,
                n_out_conv,
            )

        elif depth > 0:

            if groups is not None:
                groups = 1 if n_out_channels < groups else groups
                self.bn0, self.bn1 = self.norm(groups, n_out_channels), self.norm(groups, n_out_channels)
            else:
                self.bn0, self.bn1 = self.norm(n_out_channels), self.norm(n_out_channels)

            self.relu0 = act_inplace(self.activation)
            self.relu1 = act_inplace(self.activation)

            n_in_conv_enc = n_in_channels
            n_out_conv_enc = n_out_channels if not self.double_down else n_in_channels

            n_in_down = n_out_channels if not self.double_down else n_in_channels
            if self.double_down:
                n_out_down = n_out_channels * 2 if self.depth == depth_parent else n_out_channels
            else:
                n_out_down = n_out_channels

            n_in_conv_dec = 2 * n_out_channels if not self.double_down else n_out_conv_enc
            n_out_conv_dec = n_out_channels if not self.double_down else n_in_conv_enc

            n_in_up = 2 * n_out_channels if not self.double_down else n_out_down
            n_out_up = n_out_channels if not self.double_down else n_in_down
            
            self.sub_2conv_more = self.block(
                n_in_conv_enc,
                n_out_conv_enc,
            )

            self.conv_down = self.downsample(n_in_down, n_out_down)
            self.convt = self.upsample(n_in_up, n_out_up)

            self.sub_2conv_less = self.block(
                n_in_conv_dec,
                n_out_conv_dec,
            )

            if self.double_down and self.depth == depth_parent:
                recurse_n_in = 2 * n_out_channels
            else:
                recurse_n_in = n_out_channels

            self.sub_u = _Net_recurse(
                recurse_n_in,
                mult_chan=2,
                depth_parent=depth_parent,
                depth=(depth - 1),
                upsample=self.upsample,
                downsample=self.downsample,
                double_down=self.double_down,
                activation_fn=self.activation,
                norm_fn=self.norm,
                groups=groups,
                block=self.block,
                mid_block=self.mid_block,
            )

    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            if self.double_down:
                x_cat = x_2conv_more + x_relu1
            else:
                x_cat = torch.cat((x_2conv_more, x_relu1), 1)
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less


class SubNet2Conv(nn.Module):
    def __init__(self, n_in, n_out, activation_fn=None, norm_fn=None, order="cna", chan_attn=None, groups=None, res_block=False):
        super().__init__()
        self.res_block = res_block
        self.order = order

        if groups is not None:
            groups = 1 if n_in < groups else groups
            self.bn1 = norm_fn(groups, n_out) if order.find('n') > order.find('c') else norm_fn(groups, n_in)
            self.bn2 = norm_fn(groups, n_out)
        else:
            self.bn1 = norm_fn(n_out) if order.find('n') > order.find('c') else norm_fn(n_in)
            self.bn2 = norm_fn(n_out)

        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.relu1 = act_inplace(activation_fn)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.relu2 = act_inplace(activation_fn)

        if res_block:
            self.res_conv = nn.Conv3d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
            if res_block == "regress":
                self.beta = nn.Parameter(torch.zeros((1, n_out, 1, 1, 1)), requires_grad=True)
                self.gamma = nn.Parameter(torch.zeros((1, n_out, 1, 1, 1)), requires_grad=True)
            if "ca" in self.res_block:
                self.ca1 = chan_attn(n_out)
                self.ca2 = chan_attn(n_out)

    def forward(self, x):

        h = x
        for o in self.order:
            if o == "c":
                h = self.conv1(h)
            elif o == "n":
                h = self.bn1(h)
            elif o == "a":
                h = self.relu1(h)

        # learned skip connection over first convolution
        if self.res_block == "regress":
            y = self.res_conv(x) + h * self.beta
        # skip connection over first convolution
        elif self.res_block == "x2":
            y = self.res_conv(x) + h
        elif self.res_block == "x2ca":
            y = self.ca1(self.res_conv(x)) + h
        # covers both no skip connection and classic skip connection
        else:
            y = h

        h = y
        for o in self.order:
            if o == "c":
                h = self.conv2(h)
            elif o == "n":
                h = self.bn2(h)
            elif o == "a":
                h = self.relu2(h)

        if not self.res_block:
            y = h
        # learned skip connection over second convolution
        elif self.res_block == "regress":
            y = y + h * self.gamma
        # skip connection over second convolution
        elif self.res_block == "x2":
            y = y + h
        elif self.res_block == "x2ca":
            y = self.ca2(y) + h
        # classic skip connection over both convolutions
        else:
            y = self.res_conv(x) + h

        return y


class DilatedBottleneck(nn.Module):
    def __init__(self, n_in, n_out, depth=4, activation_fn=None, norm_fn=None, order="cna", chan_attn=None, groups=None):
        super().__init__()
        self.order = order

        activation = act_inplace(activation_fn)
        if groups is not None:
            groups = 1 if n_in < groups else groups
            norm = norm_fn(groups, n_out) if order.find('n') > order.find('c') else norm_fn(groups, n_in)
        else:
            norm = norm_fn(n_out) if order.find('n') > order.find('c') else norm_fn(n_in)

        for i in range(depth):
            dilate = 2**i
            model = []
            for o in order:
                if o == "c":
                    model.append(nn.Conv3d(n_in, n_out, kernel_size=3, padding=dilate, dilation=dilate))
                elif o == "n":
                    model.append(norm)
                elif o == "a":
                    model.append(activation)
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
        activation_fn = activation_fn or nn.Identity

        self.net = nn.Sequential(
            conv,
            activation_fn(),
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
    

class PlainBaselineBlock(nn.Module):
    def __init__(self, n_in, n_out, dw_expand=1, ffn_expand=2, chan_attn=None, activation_fn=None, norm_fn=None, groups=None):
        super().__init__()
        norm_fn = default(norm_fn, "identity")

        if activation_fn is not None:
            self.activ1 = act_inplace(get_act(activation_fn))
            self.activ2 = act_inplace(get_act(activation_fn))
        else:
            self.activ1 = nn.ReLU()
            self.activ2 = nn.ReLU()

        if groups is not None:
            groups = 1 if n_in < groups else groups
            self.norm1 = get_norm(norm_fn)(groups, n_out)
            self.norm2 = get_norm(norm_fn)(groups, n_out)
        else:
            self.norm1 = get_norm(norm_fn)(n_out)
            self.norm2 = get_norm(norm_fn)(n_out)

        n_dw = n_in * dw_expand
        self.conv1 = nn.Conv3d(n_in, n_dw, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=n_dw, out_channels=n_dw, kernel_size=3, padding=1, stride=1, groups=n_dw, bias=True)
        self.conv3 = nn.Conv3d(in_channels=n_dw, out_channels=n_in, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.ca = get_ca(chan_attn, n_dw)
        
        n_ffn = n_in * ffn_expand
        self.conv4 = nn.Conv3d(in_channels=n_in, out_channels=n_ffn, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv3d(in_channels=n_ffn, out_channels=n_in, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, n_in, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, n_in, 1, 1, 1)), requires_grad=True)


    def forward(self, x):

        h = self.norm1(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.activ1(h)
        h = h * self.ca(h)
        h = self.conv3(h)

        y = x + h * self.beta

        h = self.norm2(y)
        h = self.conv4(h)
        h = self.activ2(h)
        h = self.conv5(h)
 
        return y + h * self.gamma
