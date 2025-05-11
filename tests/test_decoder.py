import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.layers import Decoder
import torch

if __name__ == "__main__":
    batch_size = 2
    tgt_seq_len = 8
    src_seq_len = 10
    vocab_size = 1000
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 4

    dummy_tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))         # Target token IDs
    dummy_enc_out = torch.rand(batch_size, src_seq_len, d_model)                # Encoder output

    decoder = Decoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    output = decoder(dummy_tgt, dummy_enc_out)

    print("Target input shape:", dummy_tgt.shape)
    print("Encoder output shape:", dummy_enc_out.shape)
    print("Decoder output shape:", output.shape)
