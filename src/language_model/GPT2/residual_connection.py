import torch
import torch.nn as nn
from src.language_model.GPT2.layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Residual connection module with layer normalization and dropout.
    To facilitate the training and improve the learning capability of deep networks.

    Args:
        config (Config): An instance of the configuration class with model hyperparameters.
    """

    def __init__(self, config):
        """
        Initializes the ResidualConnection module.

        Parameters:
        - config (Config): An instance of the configuration class with model hyperparameters.
        """
        super(ResidualConnection, self).__init__()
        self.config = config
        self.ln = LayerNormalization(self.config)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, sublayer, is_attention=False):
        """
        Forward pass of the ResidualConnection module.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - sublayer (nn.Module): Sublayer module applied within the residual connection.
        - is_attention (bool, optional): Whether the sublayer is an attention layer.

        Returns:
        - torch.Tensor or Tuple: Output tensor after applying the residual connection.
          If is_attention is True, returns a tuple (x, present, output_attentions).
          Shape: (batch_size, max_seq_len, d_model)
        """
        if is_attention:
            outputs = sublayer(self.ln(x))
            x = x + self.dropout(outputs[0])
            # return the same outputs as the attention layer
            outputs = (x,) + outputs[1:]
            return outputs  # tuple (x, present, output_attentions)
        return x + self.dropout(sublayer(self.ln(x)))


if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    config.d_model = 512
    config.dropout = 0.1
    rc = ResidualConnection(config)
    x = torch.randn(2, 5, config.d_model)
    print(rc(x, lambda x: x).size()) # torch.Size([2, 5, 512])
    print(rc)