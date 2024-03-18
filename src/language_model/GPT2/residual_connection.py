import torch
import torch.nn as nn
import torch.nn.functional as F
from src.language_model.GPT2.layer_normalization import LayerNormalization
import math

class ResidualConnection(nn.Module):
    def __init__(self, config):
        super(ResidualConnection, self).__init__()
        self.config = config
        self.ln = LayerNormalization(self.config)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, sublayer,is_attention=False):
        if is_attention:
            outputs = sublayer(self.ln(x))
            x = x + self.dropout(outputs[0])
            # return same outputs as the attention layer
            outputs = (x,) + outputs[1:]
            return outputs # tuple (x, present, output_attentions)
        return x + self.dropout(sublayer(self.ln(x))) # (batch_size, max_seq_len, d_model)

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