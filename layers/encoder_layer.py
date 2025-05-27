import torch
import torch.nn as nn
from attention.multi_head_attention import MultiHeadAttention
from layers.feed_forward import FeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.self_attention=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=FeedForwardNetwork(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        attn_output,_=self.self_attention(x,x,x,mask)
        x=self.norm1(x+self.dropout1(attn_output))

        #feed forward
        ff_output=self.feed_forward(x)
        x=self.norm2(x+self.dropout2(ff_output))
        return x