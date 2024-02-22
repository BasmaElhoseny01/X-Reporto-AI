from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    """
    Module responsible for embedding input tokens and applying dropout.

    Args:
        config (Config): An instance of the configuration class with model hyperparameters.
    """

    def __init__(self, config):
        """
        Initializes the InputEmbedding module.

        Parameters:
        - config (Config): An instance of the configuration class with model hyperparameters.
        """
        super(InputEmbedding, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x:Optional[torch.LongTensor]):
        """
        Forward pass of the InputEmbedding module.

        Parameters:
        - x (torch.Tensor): Input tensor representing token indices.

        Returns:
        - torch.Tensor: Output tensor after token embedding and dropout.
          Shape: (batch_size, max_seq_len, d_model)
        """
        """
        Some Explaintion:
        1- self.token_embedding(x) * math.sqrt(self.config.d_model):
        The idea is to maintain a reasonable scale for the inputs to the transformer model, which can help in stabilizing the training process.
        2- Dropout is to prevent overfitting and enhance the generalization ability of the model.
        """
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        return self.dropout(x)
