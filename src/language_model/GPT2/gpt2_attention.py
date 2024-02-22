from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.language_model.GPT2.layer_normalization import LayerNormalization
from src.language_model.GPT2.feed_forward import FeedForward
from src.language_model.GPT2.positional_encoding import PositionalEncoding
from src.language_model.GPT2.residual_connection import ResidualConnection
from src.language_model.GPT2.conv1d import Conv1D
import sys
import math

class CustomGPT2MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(CustomGPT2MultiHeadAttention, self).__init__()
        self.config = config
        self.d_model = self.config.d_model
        self.num_heads = self.config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.max_seq_len = self.config.max_seq_len
        
        # create causal mask for self-attention of shape (1, 1, max_seq_len+1, max_seq_len+1) as trianlge matrix of ones at below diagonal
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((self.max_seq_len+1, self.max_seq_len+1), dtype=torch.bool)).view(
                1, 1, self.max_seq_len+1, self.max_seq_len+1
            ),
            persistent=False,
        )

        # create mask_value for self-attention of shape (1)
        self.register_buffer("mask_value", torch.tensor(-1e4), persistent=False)

        #Test Conv1D instead of Linear
        self.u_k = Conv1D(self.d_model, self.d_model) # attention key of image visual feature
        self.u_v = Conv1D(self.d_model, self.d_model) # attention value of image visual feature

        self.c_attn = Conv1D(3 * self.d_model, self.d_model) # attention key, query, value of text feature
        self.c_proj = Conv1D(self.d_model, self.d_model) # attention output projection

        self.dropout = nn.Dropout(config.dropout)   

        # assert with print "assert self.d_model % self.num_heads == 0"
        assert self.d_model % self.num_heads == 0 , "d_model should be divisible by num_heads"

    @staticmethod
    def pesudo_attention(query, key, value,causal_mask,mask_value, mask=None, dropout=None):
        # query, key, value: (batch_size, h, max_seq_len, d_k)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1)) # (batch_size, h, max_seq_len, max_seq_len+1)

        query_length, key_length = query.size(-2), key.size(-2) 

        # in training, key_length = 1025, query_length = 1024 , causal_mask_selected = (1, 1, 1024, 1025)
        causal_mask_selected = causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        scores = torch.where(causal_mask_selected, scores.to(scores.dtype), mask_value) # (batch_size, h, max_seq_len, max_seq_len+1)

        if mask is not None:
            # add mask to scores
            scores = scores + mask
        
        # apply softmax to scores
        p_attn = F.softmax(scores, dim=-1) # (batch_size, h, max_seq_len, max_seq_len+1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        # apply attention to value
        return torch.matmul(p_attn, value), p_attn # (batch_size, h, max_seq_len, d_k), (batch_size, h, max_seq_len, max_seq_len+1)
    def forward(self,
                hidden_states: Optional[Tuple[torch.FloatTensor]],
                attention_mask: Optional[torch.FloatTensor] = None,
                image_hidden_states: Optional[torch.Tensor] = None, # (batch_size, 1, d_model)
                layer_past: Optional[Tuple[torch.Tensor]] = None, #(key, value) for past sequence
                use_cache: Optional[bool] = False, # whether to use cache for decoding
                output_attentions: Optional[bool] = False,):
        batch_size = hidden_states.size(0)
        
        q, k, v = self.c_attn(hidden_states).split(self.d_model, dim=2)
        q = q.view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        v = v.view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        
        # if first layer: add image feature to attention
        if image_hidden_states is not None and layer_past is None:
            k_image = self.u_k(image_hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, 1, d_model//num_heads)
            v_image = self.u_v(image_hidden_states).view(batch_size, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2) # (batch_size, num_heads, 1, d_model//num_heads)
        
            # concat k, v from image and text on dim=2
            k = torch.cat((k_image, k), dim=2)
            v = torch.cat((v_image, v), dim=2)

        # concat k, v from past and current on dim=2
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)  # (batch_size, num_heads, past_seq_len + seq_len, d_model//num_heads)
            v = torch.cat((past_v, v), dim=-2)  # (batch_size, num_heads, past_seq_len + seq_len, d_model//num_heads)

        # if generation mode use cache to get past_k, past_v
        if use_cache is True:
            present = (k, v)
        else:
            present = None

        # calculate attention and apply to value
        x, attn = CustomGPT2MultiHeadAttention.pesudo_attention(q, k, v,self.causal_mask,self.mask_value, attention_mask, self.dropout) # (batch_size, num_heads, max_seq_len, d_model//num_heads), (batch_size, num_heads, max_seq_len, max_seq_len+1)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, max_seq_len, d_model) 
        x = self.c_proj(x)
        outputs = (x, present)

        if output_attentions:
            outputs += (attn,)
        return outputs # a (batch_size, max_seq_len, d_model), present, (attentions) (batch_size, num_heads, max_seq_len, max_seq_len+1)
    