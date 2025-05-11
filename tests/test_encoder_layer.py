import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.layers import EncoderLayer
import torch

if __name__ == "__main__":
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8
    d_ff = 256

    dummy_input = torch.rand(batch_size, seq_len, d_model)

    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    output = encoder_layer(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Encoder output shape:", output.shape)
