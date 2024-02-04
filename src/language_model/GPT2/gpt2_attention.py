from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.language_model.GPT2.layer_normalization import LayerNormalization
from src.language_model.GPT2.feed_forward import FeedForward
from src.language_model.GPT2.positional_encoding import PositionalEncoding
from src.language_model.GPT2.residual_connection import ResidualConnection
import math

class CustomGPT2MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(CustomGPT2MultiHeadAttention, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        
        # hidden state to query, key, value
        self.w_q = nn.Linear(self.d_model, self.d_model,bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model,bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model,bias=False)
        
        # image hidden state to key, value
        self.u_k = nn.Linear(self.d_model, self.d_model,bias=False)
        self.u_v = nn.Linear(self.d_model, self.d_model,bias=False)

        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False) # Wo
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNormalization(config)

        # assert with print "assert self.d_model % self.num_heads == 0"
        assert self.d_model % self.num_heads == 0 , "d_model should be divisible by num_heads"

    @staticmethod
    def pesudo_attention(query, key, value, mask=None, dropout=None):
        # query, key, value: (batch_size, h, max_seq_len, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1) # (batch_size, h, max_seq_len, max_seq_len)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn # (batch_size, h, max_seq_len, d_k), (batch_size, h, max_seq_len, max_seq_len)
    def forward(self,
                hidden_states: Optional[Tuple[torch.FloatTensor]],
                layer_past: Optional[Tuple[torch.Tensor]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                image_hidden_states: Optional[torch.Tensor] = None, # (batch_size, 1, d_model)
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,):
        batch_size = hidden_states.size(0)
        q = self.w_q(hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        k = self.w_k(hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        v = self.w_v(hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        if image_hidden_states is not None:
            k_image = self.u_k(image_hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, 1, d_model//num_heads)
            v_image = self.u_v(image_hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, 1, d_model//num_heads)
        
        # concat k, v from image and text on dim=2
        if image_hidden_states is not None:
            k = torch.cat((k, k_image), dim=2) # (batch_size, num_heads, max_seq_len+1, d_model//num_heads)
            v = torch.cat((v, v_image), dim=2) # (batch_size, num_heads, max_seq_len+1, d_model//num_heads)

        x, attn = self.pesudo_attention(q, k, v, attention_mask, self.dropout) # (batch_size, num_heads, max_seq_len, d_model//num_heads), (batch_size, num_heads, max_seq_len, max_seq_len+1)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, max_seq_len, d_model) 
        x = self.w_o(x) # (batch_size, max_seq_len, d_model)
        if output_attentions:
            return x, attn # (batch_size, max_seq_len, d_model), (batch_size, num_heads, max_seq_len, max_seq_len+1)
        return x # (batch_size, max_seq_len, d_model), (batch_size, num_heads, max_seq_len, max_seq_len+1)
    