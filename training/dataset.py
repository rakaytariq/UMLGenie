import torch
from torch.utils.data import Dataset
import json

class UMLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_input_len=128, max_output_len=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def pad(self, ids, max_len):
        pad_id = self.tokenizer.word2idx["<pad>"]
        return ids[:max_len] + [pad_id] * max(0, max_len - len(ids))

    def __getitem__(self, idx):
        example = self.data[idx]
        input_ids = self.tokenizer.encode(example["input_text"])
        output_ids = self.tokenizer.encode(example["output_text"])

        input_ids = self.pad(input_ids, self.max_input_len)
        output_ids = self.pad(output_ids, self.max_output_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long)
        }
