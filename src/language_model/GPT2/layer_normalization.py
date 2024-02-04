import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNormalization(nn.Module):
    def __init__(self, config):
        super(LayerNormalization, self).__init__()
        self.config = config
        self.gamma = nn.Parameter(torch.ones(self.config.d_model))
        self.beta = nn.Parameter(torch.zeros(self.config.d_model))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # (batch_size, max_seq_len, 1)
        std = x.std(-1, keepdim=True) # (batch_size, max_seq_len, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta # (batch_size, max_seq_len, d_model)
    
if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    config.d_model = 512
    ln = LayerNormalization(config)
    x = torch.randn(2, 5, config.d_model)
    print(ln(x).size()) # torch.Size([2, 5, 512])
    print(ln)