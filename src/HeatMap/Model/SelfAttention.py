import torch
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,dropout=0.1, activation="relu"):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Implementation of LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()


    def forward(self,src, src_mask= None, src_key_padding_mask = None):
        # multihead attention
        src2, attn_weights = self.attention(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # add & norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # add & norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class SelfAttention(nn.Module):
    def __init__(self,  d_model, nhead = 4,dropout=0.1):
        super(SelfAttention, self).__init__()
        self.encoder_layer = TransformerEncoder(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
    
    def forward(self, k,mask=None):
        atten=None
        k=k.transpose(0,1)
        k, atten = self.encoder_layer(k, src_mask=mask)
        k=k.transpose(0,1)
        return k, atten

