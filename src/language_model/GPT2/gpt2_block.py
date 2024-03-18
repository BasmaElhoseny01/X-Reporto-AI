import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.language_model.GPT2.embeddings import InputEmbedding
from src.language_model.GPT2.positional_encoding import PositionalEncoding
from src.language_model.GPT2.feed_forward import FeedForward
from src.language_model.GPT2.residual_connection import ResidualConnection
from src.language_model.GPT2.layer_normalization import LayerNormalization
from src.language_model.GPT2.gpt2_attention import CustomGPT2MultiHeadAttention

class CustomGPT2Block(nn.Module):
    def __init__(self, config):
        super(CustomGPT2Block, self).__init__()
        self.config = config
        self.rc1 = ResidualConnection(self.config)
        self.attn = CustomGPT2MultiHeadAttention(self.config)
        self.rc2 = ResidualConnection(self.config)
        self.ff = FeedForward(self.config)

    def forward(self, x,attention_mask, image_hidden_states=None):
        x = self.rc1(x, lambda x: self.attn(x, image_hidden_states=image_hidden_states,attention_mask=attention_mask))
        x = self.rc2(x, lambda x: self.ff(x))
        return x

if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    # config.d_model = 512
    # config.d_ff = 2048
    # config.num_heads = 8
    # config.dropout = 0.1
    gpt2_block = CustomGPT2Block(config)
    x = torch.randn(2, 5, config.d_model)
    attention_mask = torch.ones(2, 5, dtype=x.dtype, device=x.device)
    attention_mask = attention_mask[:, None, None, :]
    # attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * -10000.0

    image_hidden_states = torch.randn(2, 1, config.d_model)
    print(gpt2_block(x,attention_mask).size()) # torch.Size([2, 5, 512])
    print(gpt2_block)
    x = torch.randn(2, 5, config.d_model)
    ones = torch.ones(attention_mask.size()[:-1] + (1,), dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask = torch.cat((ones, attention_mask), dim=-1)
    # attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * -10000.0
    image_hidden_states = torch.randn(2, 1, config.d_model)
    print(gpt2_block(x,attention_mask, image_hidden_states).size()) # torch.Size([2, 5, 512])
    print(gpt2_block)
    