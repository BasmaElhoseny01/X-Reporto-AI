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

    def forward(self, x,attention_mask, image_hidden_states=None,layer_past = None,use_cache = False,output_attentions = False):
        # apply residual connection to attention layer
        outputs = self.rc1(x, lambda x: self.attn(x, image_hidden_states=image_hidden_states,attention_mask=attention_mask
                                            ,layer_past = layer_past,use_cache = use_cache,output_attentions = output_attentions),is_attention=True)
        x = outputs[0]  # output of attention layer
        outputs = outputs[1:] # present, output_attentions

        # apply residual connection to feed forward layer
        x = self.rc2(x, lambda x: self.ff(x))
        if output_attentions is False:
            outputs = (outputs[0],) # tuple (present,)
        if use_cache:
            outputs = (x,) + outputs[0:]
            return outputs # tuple (x, present, output_attentions)
        return (x,) + outputs[1:] # tuple (x, output_attentions)

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
    print(gpt2_block(x,attention_mask)[0].size()) # torch.Size([2, 5, 512])
    print(gpt2_block)
    x = torch.randn(2, 5, config.d_model)
    ones = torch.ones(attention_mask.size()[:-1] + (1,), dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask = torch.cat((ones, attention_mask), dim=-1)
    # attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * -10000.0
    image_hidden_states = torch.randn(2, 1, config.d_model)
    print(gpt2_block(x,attention_mask, image_hidden_states)[0].size()) # torch.Size([2, 5, 512])
    print(gpt2_block)
    