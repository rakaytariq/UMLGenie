import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.layers import ScaledDotProductAttention, MultiHeadAttention
import torch

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8

    dummy_query = torch.rand(batch_size, seq_len, d_model)
    dummy_key = torch.rand(batch_size, seq_len, d_model)
    dummy_value = torch.rand(batch_size, seq_len, d_model)

    print("Testing Scaled Dot-Product Attention")
    attention = ScaledDotProductAttention()
    output, attn_weights = attention(dummy_query, dummy_key, dummy_value)
    print("Output shape:", output.shape)
    print("Attention shape:", attn_weights.shape)

    print("\nTesting Multi-Head Attention")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output = mha(dummy_query, dummy_key, dummy_value)
    print("Multi-head output shape:", mha_output.shape)
