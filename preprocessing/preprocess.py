from datasets import load_dataset
import json
import os

def load_raw_dataset():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("ReshmaUMLGraphMaster/Usecase-UML")
    return dataset

def convert_to_io_pairs(example):
    """
    Converts one data point into input/output text.
    """
    input_text = example["prompt"]           # user story
    output_text = example["response"]        # already a JSON string or formatted text
    return {"input_text": input_text, "output_text": output_text}

def preprocess_dataset(dataset):
    print("Preprocessing dataset...")
    return dataset.map(convert_to_io_pairs)

def save_dataset(dataset, split="train", path="data/processed"):
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, f"{split}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Saved preprocessed {split} split to {output_path}")
