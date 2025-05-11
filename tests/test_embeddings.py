import sys
import os

# Ensure parent directory (project root) is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.embeddings import TokenEmbedding, PositionalEncoding
import torch

if __name__ == "__main__":
    vocab_size = 100
    d_model = 64
    seq_len = 10
    batch_size = 2

    token_emb = TokenEmbedding(vocab_size, d_model)
    pos_enc = PositionalEncoding(d_model)

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedded = token_emb(dummy_input)
    out = pos_enc(embedded)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", out.shape)
