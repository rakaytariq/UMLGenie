import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer import Transformer
import torch

if __name__ == "__main__":
    batch_size = 2
    src_seq_len = 12
    tgt_seq_len = 10
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    num_heads = 8
    d_ff = 512
    num_layers = 4

    dummy_src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    dummy_tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    logits = model(dummy_src, dummy_tgt)

    print("Input (src) shape:", dummy_src.shape)
    print("Target (tgt) shape:", dummy_tgt.shape)
    print("Model output (logits) shape:", logits.shape)
