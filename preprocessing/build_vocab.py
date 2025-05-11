from tokenizer import Tokenizer
import json
import os

def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return [d["input_text"] for d in data] + [d["output_text"] for d in data]

if __name__ == "__main__":
    input_file = "data/processed_dsl/train.jsonl"
    save_path = "data/vocab/vocab.json"

    texts = load_texts(input_file)

    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts, max_size=5000, min_freq=1)
    tokenizer.save(save_path)

    print(f"✅ Vocabulary saved to {save_path}")
    print("First 10 tokens:", tokenizer.vocab[:10])
    print("<eos> in vocab?", "<eos>" in tokenizer.vocab)
