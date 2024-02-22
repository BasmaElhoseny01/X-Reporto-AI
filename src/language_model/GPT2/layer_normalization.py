import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Layer normalization module for normalizing the activations of a neural network layer.
    Layer normalization ensures “all neurons in a particular layer effectively have the same distribution across all features for a given input.”

    Args:
        config (Config): An instance of the configuration class with model hyperparameters.
    """

    def __init__(self, config):
        """
        Initializes the LayerNormalization module.

        Parameters:
        - config (Config): An instance of the configuration class with model hyperparameters.
        """
        super(LayerNormalization, self).__init__()
        self.config = config
        self.gamma = nn.Parameter(torch.ones(self.config.d_model))
        self.beta = nn.Parameter(torch.zeros(self.config.d_model))
        self.eps = 1e-05

    def forward(self, x):
        """
        Forward pass of the LayerNormalization module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying layer normalization.
          Shape: (batch_size, max_seq_len, d_model)
        """
        """
        Some Explaintion:
        """
        mean = x.mean(-1, keepdim=True)  # (batch_size, max_seq_len, 1)
        std = x.std(-1, keepdim=True)  # (batch_size, max_seq_len, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  # (batch_size, max_seq_len, d_model)

    
if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    config.d_model = 512
    ln = LayerNormalization(config)
    x = torch.randn(2, 5, config.d_model)
    print(ln(x).size()) # torch.Size([2, 5, 512])
    print(ln)