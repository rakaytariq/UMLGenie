import torch
import torch.nn as nn
from model.layers import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        dropout=0.1,
        max_len=512
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        output = self.output_proj(dec_output)
        return output  # shape: [batch_size, tgt_len, tgt_vocab_size]
