import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.config = config
        self.register_buffer('positional_encoding', self._get_positional_encoding())
        self.dropout = nn.Dropout(self.config.dropout)
        # self.wpe = nn.Embedding(self.config.max_position_embeddings, self.config.d_model)


    def _get_positional_encoding(self):
        pe = torch.zeros(self.config.max_seq_len, self.config.d_model) # (max_seq_len, d_model)
        position = torch.arange(0, self.config.max_seq_len, dtype=torch.float).unsqueeze(1) # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2).float() * (-math.log(10000.0) / self.config.d_model)) # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        self.register_buffer('positional_encoding', pe)
        return pe # (1, max_seq_len, d_model)

    def forward(self, x,past_length=None):
        if past_length is not None:
            x = x + (self.positional_encoding[:, past_length-1:past_length+x.size(1)-1,:]).requires_grad_(False)
        else:
            x = x + (self.positional_encoding[:, :x.size(1),:]).requires_grad_(False)
        # x = x + self.wpe(torch.arange(x.size(1)).to(x.device)).unsqueeze(0)
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