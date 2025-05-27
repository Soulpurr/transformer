import torch
import torch.nn as nn
from attention.multi_head_attention import MultiHeadAttention
from layers.feed_forward import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super(DecoderLayer,self).__init__()

        self.self_attention=MultiHeadAttention(d_model,num_heads)
        self.enc_dec_attention=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=FeedForwardNetwork(d_model,d_ff,dropout)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 1. Masked self-attention
        attn1, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn1))

        # 2. Encoder-decoder attention (cross-attention)
        attn2, _ = self.enc_dec_attention(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout2(attn2))

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

