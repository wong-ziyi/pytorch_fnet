import fnet.nn_modules.fnet_nn_3d_params


class Net(fnet.nn_modules.fnet_nn_3d_params.Net):
    def __init__(self, depth=4, mult_chan=32):
        super().__init__(depth=depth, mult_chan=mult_chan)
