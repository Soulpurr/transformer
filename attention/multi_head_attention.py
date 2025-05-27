import torch
import torch.nn as nn
from .scaled_dot_product import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_head):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_head==0, "d_model must be divisible by num_heads"
        self.d_model=d_model #dmodel is full embedding size (eg 512)
        self.num_head=num_head
        self.depth=d_model//num_head #each head's dimension
        # each head gets a chunk of size depth

        #Linear layers for projecting input to q,k,v
        self.wq=nn.Linear(d_model,d_model)
        self.wk=nn.Linear(d_model,d_model)
        self.wv=nn.Linear(d_model,d_model)

        #op linear layer after concatenation
        self.dense=nn.Linear(d_model,d_model)
    def split_heads(self,x,batch_size):
        """
            Split the last dimenension into (num_head,depth)
            Transpose the result to shape (batch_size,num_heads,seq_len,depth)

        """
        x=x.view(batch_size,-1,self.num_head,self.depth)
        return x.transpose(1,2)
    def forward(self,q,k,v,mask=None):
        batch_size=q.size(0)

        #Linear Projections
        q=self.wq(q) #(B,s_q,d_model)
        k=self.wk(k) #(B,s_k,d_model)
        v=self.wv(v) #(B,s_v,d_model)

        #split them to multiple heads
        q=self.split_heads(q,batch_size)
        k=self.split_heads(k,batch_size)
        v=self.split_heads(v,batch_size)
       

        #scaled dot product
        scaled_attention,attention_weights=scaled_dot_product_attention(q,k,v,mask)

        #concatenate heads
        scaled_attention=scaled_attention.transpose(1,2) #(B,s_q,H,depth)
        concat_attention=scaled_attention.reshape(batch_size,-1,self.d_model)

        #final layer
        op=self.dense(concat_attention)
        return op,attention_weights
