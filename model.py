##Input embedding
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #Create a matrix of shape (seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)
        #Formula for positional encoding(simplified one)
        #Create a vector of shape (seqlen)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        ##Apply sin to even position
        pe[:,0::2]=torch.sin(position*div_term)
        ##Apply cos to odd position
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
    
class FeedForwardLayer(nn.torch):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)
    def forward(self,x):
       #(Batch,seq_len,d_model)->(Batch,seq_len,d_ff)->(Batch,seq_len,d_model) 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.torch):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0, "d_model is not divisible by h"
        self.d_k=d_model//h

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_0=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))//math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores

    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        # (batch,seqlen,dmodel)-->(batch,seqlen,h,d_k)-->(batch,h,seqlen,,d_k)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self_attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_0(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float)->None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardLayer,dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList)->None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)   
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardLayer,dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList)->None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)   

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)









