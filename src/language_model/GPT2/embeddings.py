import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self, config):
        super(InputEmbedding, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model) # (batch_size, max_seq_len, d_model)
        return self.dropout(x) # (batch_size, max_seq_len, d_model)

if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    config.vocab_size = 100
    config.d_model = 512
    config.dropout = 0.1
    ie = InputEmbedding(config)
    x = torch.randint(0, 100, (2, 5))
    print(ie(x).size()) # torch.Size([2, 5, 512])
    print(ie)