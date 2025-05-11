from preprocessing.preprocess import load_raw_dataset, preprocess_dataset, save_dataset
from datasets import DatasetDict

if __name__ == "__main__":
    raw = load_raw_dataset()

    # Only 'train' exists, so manually split it
    full_dataset = raw["train"].train_test_split(test_size=0.2, seed=42)
    test_valid = full_dataset["test"].train_test_split(test_size=0.5, seed=42)

    split_dataset = DatasetDict({
        "train": full_dataset["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"]
    })

    for split in split_dataset:
        processed = preprocess_dataset(split_dataset[split])
        save_dataset(processed, split)
