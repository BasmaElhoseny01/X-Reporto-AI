import torch
import torch.nn as nn

class Conv1D(nn.Module):
    """
    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
    """

    def __init__(self, nf, nx):
        """
        Initializes a 1D-convolutional layer.

        Parameters:
        - nf (int): The number of output features.
        - nx (int): The number of input features.
        """
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        """
        Forward pass of the 1D-convolutional layer.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying the convolutional layer.
        """
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
