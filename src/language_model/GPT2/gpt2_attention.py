from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.language_model.GPT2.conv1d import Conv1D
import math
import sys
class CustomGPT2MultiHeadAttention(nn.Module):
    """
    Custom multi-head attention layer for GPT-2 model.

    Args:
        config (Config): An instance of the configuration class with model hyperparameters.
    """

    def __init__(self, config):
        """
        Initializes the CustomGPT2MultiHeadAttention.

        Parameters:
        - config (Config): An instance of the configuration class with model hyperparameters.
        """
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

        # Mask value for self-attention of shape (1)
        self.register_buffer("mask_value", torch.tensor(-1e4), persistent=False)

        # Attention key and value for image visual feature
        self.u_k = Conv1D(self.d_model, self.d_model)
        self.u_v = Conv1D(self.d_model, self.d_model)

        # Attention key, query, value for text feature
        self.c_attn = Conv1D(3 * self.d_model, self.d_model)
        self.c_proj = Conv1D(self.d_model, self.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Check if d_model is divisible by num_heads
        assert self.d_model % self.num_heads == 0, "d_model should be divisible by num_heads"

    @staticmethod
    def pesudo_attention(query, key, value, causal_mask, mask_value, mask=None, dropout=None):
        """
        Pseudo-attention function used in the GPT-2 multi-head attention.

        Parameters:
        - query, key, value (torch.Tensor): Input tensors, (batch_size, h, max_seq_len, d_k)

        - causal_mask (torch.Tensor): Causal mask for self-attention.
        - mask_value (torch.Tensor): Mask value for self-attention.
        - mask (torch.Tensor, optional): Additional mask for attention.
        - dropout (nn.Dropout, optional): Dropout layer.

        Returns:
        - Tuple[torch.Tensor]: Output tensor after applying pseudo-attention and attention probabilities.
        """
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))  # (batch_size, h, max_seq_len, max_seq_len+1)
        query_length, key_length = query.size(-2), key.size(-2)
        # in training, key_length = 1025, query_length = 1024 , causal_mask_selected = (1, 1, 1024, 1025)
        causal_mask_selected = causal_mask[:, :, key_length - query_length : key_length, :key_length]
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        
        scores = torch.where(causal_mask_selected, scores.to(scores.dtype), mask_value) # (batch_size, h, max_seq_len, max_seq_len+1)

        if mask is not None:
            scores = scores + mask
            # apply softmax to scores
        p_attn = F.softmax(scores, dim=-1)  # (batch_size, h, max_seq_len, max_seq_len+1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        # apply attention to value
        return torch.matmul(p_attn, value), p_attn  # (batch_size, h, max_seq_len, d_k), (batch_size, h, max_seq_len, max_seq_len+1)
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,  # (batch_size, 1, d_model)
        layer_past: Optional[Tuple[torch.Tensor]] = None,   # (key, value) for past sequence
        use_cache: Optional[bool] = False,  # whether to use cache for decoding
        output_attentions: Optional[bool] = False,
    ):
        """
        Forward pass of the CustomGPT2MultiHeadAttention.

        Parameters:
        - hidden_states (Optional[Tuple[torch.FloatTensor]]): Input tensor.
        - attention_mask (Optional[torch.FloatTensor]): Attention mask for masking out padded tokens.
        - image_hidden_states (Optional[torch.Tensor]): Hidden states from image processing.
        - layer_past (Optional[Tuple[torch.Tensor]]): Cached past attention weights.
        - use_cache (Optional[bool]): Whether to use caching for attention weights.
        - output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
        - Tuple[torch.Tensor]: Output tensor, present states (if use_cache), and attention weights (if output_attentions).
        """
        batch_size = hidden_states.size(0)

        q, k, v = self.c_attn(hidden_states).split(self.d_model, dim=2)
        q = q.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, max_seq_len, d_model//num_heads)
        v = v.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, max_seq_len, d_model//num_heads)

        if image_hidden_states is not None and layer_past is None:
            k_image = self.u_k(image_hidden_states).view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
            v_image = self.u_v(image_hidden_states).view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
            k = torch.cat((k_image, k), dim=2)
            v = torch.cat((v_image, v), dim=2)
            # print("k_image.shape", k_image.shape)

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
        x, attn = CustomGPT2MultiHeadAttention.pesudo_attention(q, k, v, self.causal_mask, self.mask_value, attention_mask, self.dropout)   # (batch_size, num_heads, max_seq_len, d_model//num_heads), (batch_size, num_heads, max_seq_len, max_seq_len+1)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)    # (batch_size, max_seq_len, d_model) 
        x = self.c_proj(x)
        outputs = (x, present)

        if output_attentions:
            outputs += (attn,)

        return outputs  # a (batch_size, max_seq_len, d_model), present, (attentions) (batch_size, num_heads, max_seq_len, max_seq_len+1)

