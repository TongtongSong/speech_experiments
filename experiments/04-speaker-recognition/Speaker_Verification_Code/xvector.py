from pooling import *
import torch
import torch.nn as nn


class Xvector(torch.nn.Module):
    """This model extracts X-vectors for speaker recognition and diarization.
    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.
    Example
    -------
    >>> compute_xvect = Xvector('cpu')
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
        in_channels=23,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    nn.BatchNorm1d(out_channels),
                ]
            )
            in_channels = tdnn_channels[block_index]

        # global average pooling
        # ==========> code <===========

        # Final linear transformation
        
        # ==========> code <===========

    def forward(self, x, lens=None):
        """Returns the x-vectors.
        Arguments
        ---------
        x : torch.Tensor
        """

        # ==========> code <===========
        pass
        # ==========> code <===========

