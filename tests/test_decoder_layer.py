import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.layers import DecoderLayer
import torch

if __name__ == "__main__":
    batch_size = 2
    tgt_seq_len = 7
    src_seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = 256

    dummy_decoder_input = torch.rand(batch_size, tgt_seq_len, d_model)
    dummy_encoder_output = torch.rand(batch_size, src_seq_len, d_model)

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
    output = decoder_layer(dummy_decoder_input, dummy_encoder_output)

    print("Decoder input shape:", dummy_decoder_input.shape)
    print("Encoder output shape:", dummy_encoder_output.shape)
    print("Decoder output shape:", output.shape)
