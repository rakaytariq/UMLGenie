from preprocessing.tokenizer import Tokenizer
from training.dataset import UMLDataset
import torch

if __name__ == "__main__":
    tokenizer = Tokenizer.load("data/vocab/vocab.json")
    dataset = UMLDataset("data/processed/train.jsonl", tokenizer)

    sample = dataset[0]
    print("Input IDs:", sample["input_ids"])
    print("Output IDs:", sample["output_ids"])
    print("Input (decoded):", tokenizer.decode(sample["input_ids"].tolist()))
