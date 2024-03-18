from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv1D(nn.Module):
    """
    Same functionality as Conv1D class of transformers.pytorch_utils but allows initialization with trained weights.

    Conv1D has the same functionality as a linear layer.
    It transforms the inputted hidden_states from shape [batch x sequence_len x hidden_dim] to [batch x sequence_len x 3*hidden_dim],
    thus allowing the retrieval of the query, key and value matrices
    """

    def __init__(self, trained_weight, trained_bias):
        super(Conv1D, self).__init__()
        self.weight = nn.Parameter( requires_grad=False)  # of shape [hidden_dim x 3*hidden_dim] for c_attn, of shape [hidden_dim x hidden_dim] for c_proj
        self.bias = nn.Parameter( requires_grad=False)  # of shape [3 * hidden_dim] for c_attn, of shape [hidden_dim] for c_proj

    def forward(self, x):  # x has shape [batch x sequence_len x hidden_dim]
        size_out = x.size()[:-1] + (self.weight.size(-1),)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x  # x has shape [batch x sequence_len x 3*hidden_dim] for c_attn, shape [batch x sequence_len x hidden_dim] for c_proj
