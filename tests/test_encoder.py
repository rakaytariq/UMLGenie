import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.layers import Encoder
import torch

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 4

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    encoder = Encoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    output = encoder(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Encoder output shape:", output.shape)
