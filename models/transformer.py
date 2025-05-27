import torch
import torch.nn as nn
from layers.encoder_layer import EncoderLayer
from layers.decoder_layer import DecoderLayer
from layers.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,vocab_size,d_model=512,num_heads=8,d_ff=2048,num_layers=6,dropout=0.1,max_len=512):
        super(Transformer,self).__init__()
        self.Embedding=nn.Embedding(vocab_size,d_model)
        self.pos_encoding=PositionalEncoding(d_model,max_len)
        self.encoder_layers=nn.ModuleList(
            [
                EncoderLayer(d_model,num_heads,d_ff,dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers=nn.ModuleList(
            [
                DecoderLayer(d_model,num_heads,d_ff,dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_linear=nn.Linear(d_model,vocab_size)
    def forward(self,src,tgt,src_mask=None,tgt_mask=None,memory_mask=None):
        """
        src:(B,src_seq_len)
        tgt:(B,tgt_seq_len)
        """

        #embedding + p.e
        src=self.pos_encoding(self.Embedding(src))
        tgt=self.pos_encoding(self.Embedding(tgt))

        #pass through encoder stack
        enc_output=src
        for layer in self.encoder_layers:
            enc_output=layer(enc_output,src_mask)
        # decoder stack
        dec_output=tgt
        for layer in self.decoder_layers:
            dec_output=layer(dec_output,enc_output,tgt_mask,memory_mask)
        #final linear layer to project vocab
        output=self.output_linear(dec_output)
        return output
