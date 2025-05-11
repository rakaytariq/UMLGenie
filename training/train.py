import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
import os


from model.transformer import Transformer
from training.dataset import UMLDataset
from preprocessing.tokenizer import Tokenizer

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    output_ids = [item["output_ids"] for item in batch]

    input_batch = pad_sequence(input_ids, batch_first=True, padding_value=0)
    output_batch = pad_sequence(output_ids, batch_first=True, padding_value=0)

    return input_batch, output_batch

def train():
    # Hyperparameters
    batch_size = 16
    epochs = 2
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & dataset
    tokenizer = Tokenizer.load("data/vocab/vocab.json")
    vocab_size = len(tokenizer.vocab)

    train_dataset = UMLDataset("data/processed_dsl/train.jsonl", tokenizer)
    val_dataset = UMLDataset("data/processed_dsl/train.jsonl", tokenizer)  # Just reuse for now


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4
    ).to(device)

    # Loss (ignore padding)
    pad_idx = tokenizer.word2idx["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        for input_ids, output_ids in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids, output_ids = input_ids.to(device), output_ids.to(device)

            tgt_inp = output_ids[:, :-1]
            tgt_out = output_ids[:, 1:]

            logits = model(input_ids, tgt_inp)
            logits = logits.reshape(-1, logits.size(-1))
            tgt_out = tgt_out.reshape(-1)

            loss = criterion(logits, tgt_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f}")

        # Optionally add validation loss

    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/transformer.pth")

    print("✅ Model saved to outputs/checkpoints/transformer.pth")

if __name__ == "__main__":
    print("Starting training...")  # <--- Add this line
    train()
