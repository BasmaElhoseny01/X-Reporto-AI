from typing import Optional
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to input sequences.

    Args:
        config (Config): An instance of the configuration class with model hyperparameters.
    """

    def __init__(self, config):
        """
        Initializes the PositionalEncoding module.

        Parameters:
        - config (Config): An instance of the configuration class with model hyperparameters.
        """
        super(PositionalEncoding, self).__init__()
        self.config = config
        self.register_buffer('positional_encoding', self._get_positional_encoding())
        self.dropout = nn.Dropout(self.config.dropout)

    def _get_positional_encoding(self):
        """
        Generates the positional encoding matrix.

        Returns:
        - torch.Tensor: Positional encoding matrix of shape (1, max_seq_len, d_model).
        """
        """
        Some Explaintion:
        P E(pos,2i) = sin(pos/100002i/dmodel)
        P E(pos,2i+1) = cos(pos/100002i/dmodel)
        """
        pe = torch.zeros(self.config.max_seq_len, self.config.d_model)  # (max_seq_len, d_model)
        position = torch.arange(0, self.config.max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2).float() * (-math.log(10000.0) / self.config.d_model))  # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('positional_encoding', pe)
        return pe  # (1, max_seq_len, d_model)

    def forward(self, x:Optional[torch.Tensor]=None, past_length:Optional[int]=None):
        """
        Forward pass of the PositionalEncoding module.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - past_length (int, optional): Length of past sequence, if provided.

        Returns:
        - torch.Tensor: Output tensor after adding positional encoding and applying dropout.
          Shape: (batch_size, max_seq_len, d_model)
        """
        if past_length is not None:
            x = x + (self.positional_encoding[:, past_length-1:past_length+x.size(1)-1,:]).requires_grad_(False)
        else:
              x = x + (self.positional_encoding[:, :x.size(1),:]).requires_grad_(False)
        return self.dropout(x) # (batch_size, max_seq_len, d_model)


if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    config.max_seq_len = 5
    config.d_model = 512
    config.dropout = 0.1
    pe = PositionalEncoding(config)
    x = torch.randn(2, 5, config.d_model)
    print(pe(x).size()) # torch.Size([2, 5, 512])
    print(pe)