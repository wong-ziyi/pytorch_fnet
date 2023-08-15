import fnet.nn_modules.fnet_nn_3d_params


class Net(fnet.nn_modules.fnet_nn_3d_params.Net):
    def __init__(
        self,
        depth=4,
        mult_chan=32,
        dilate=1,
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
        init_kernel_size=None,
        block_kwargs=None,
    ):
        super().__init__(
            depth=depth,
            mult_chan=mult_chan,
            dilate=dilate,
            upsample=upsample,
            downsample=downsample,
            double_down=double_down,
            activation=activation,
            norm=norm,
            order=order,
            chan_attn=chan_attn,
            groups=groups,
            block=block,
            mid_block=mid_block,
            res_block=res_block,
            init_embed=init_embed,
            init_kernel_size=init_kernel_size,
            block_kwargs=block_kwargs,
        )
