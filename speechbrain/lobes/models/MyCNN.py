"""A popular speaker recognition and diarization model.

Authors
 * Hwidong Na 2020
"""
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv2d, Conv1d
from speechbrain.nnet.normalization import BatchNorm2d, BatchNorm1d
from speechbrain.nnet.pooling import Pooling2d
from speechbrain.nnet.linear import Linear


class CNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
        The groups size of the TDNN blocks.
    scale: int
        The number of conv2d layers.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        activation=nn.LeakyReLU,
        using_pool=True,
        pooling_size=2,
        groups=1,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm2d(input_size=out_channels)
        self.pool = Pooling2d(
            pool_type="max",
            kernel_size=(pooling_size, pooling_size),
            pool_axis=(1, 2)
        )

    def forward(self, x):
        return self.pool(self.norm(self.activation(self.conv(x))))

class CNNBlock_no_Maxpool(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        activation=nn.LeakyReLU,
        groups=1,
    ):  
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm2d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))



class MyCNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=128,
        num_layers = 5,
        activation=torch.nn.ReLU,
        channels=[32, 64, 128, 256, 512],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        pooling_sizes=[2, 2, 2, 2, 1],
        groups=[1, 1, 1, 1, 1],
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial CNN layer
        for i in range(num_layers):
            if i==0:
                self.blocks.append(
                    CNNBlock(
                        in_channels = 1,
                        out_channels = channels[i],
                        kernel_size = kernel_sizes[i],
                        dilation = dilations[i],
                        pooling_size = pooling_sizes[i],
                        activation = activation,
                        groups = groups[i],
                    )
                )
            elif i!=num_layers-1:
                self.blocks.append(
                    CNNBlock(
                        in_channels = channels[i-1],
                        out_channels = channels[i],
                        kernel_size = kernel_sizes[i],
                        dilation = dilations[i],
                        pooling_size = pooling_sizes[i],
                        activation = activation,
                        groups = groups[i],
                    )   
                )
            else:
                self.blocks.append(
                    CNNBlock_no_Maxpool(
                        in_channels = channels[i-1],
                        out_channels = channels[i],
                        kernel_size = kernel_sizes[i],
                        dilation = dilations[i],
                        activation = activation,
                        groups = groups[i],
                    )
                )

        
        self.asp_bn = BatchNorm2d(input_size=channels[-1])
        self.AEP = nn.AdaptiveAvgPool2d((1,1))

        # Final linear tr ansformation
        self.fc = Conv1d(
            in_channels=channels[-1],
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch_size, time, feat, channel).
        """
        # Minimize transpose for efficiency
        x = x.unsqueeze(3)
        # x = x.transpose(1, 3)
        
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)
       
        x = x.transpose(1, -1)
        x = self.AEP(x)     # (batch_size, channel, feat, time)
        x = x.reshape((x.size(0), -1, x.size(2)*x.size(3)))
        #x = x.squeeze(-1).transpose(1,2)
        x = x.transpose(1,2)
        
        # Final linear transformation
        x = self.fc(x)
         
        # x = x.transpose(1, 2)
        return x


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=128,
        out_neurons=4,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        # x.shape:(batch_size, 1, lin_neurons)
        for layer in self.blocks:
            x = layer(x)
        
        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)
