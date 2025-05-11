import json
from datasets import load_dataset
import os

def json_to_dsl(response_str):
    try:
        if not response_str.strip():
            return None  # empty
        data = json.loads(response_str)
        actors = data.get("actors", [])
        use_cases = data.get("use_cases", [])
        relationships = data.get("relationships", [])

        lines = []
        for actor in actors:
            lines.append(f"Actor: {actor}")
        for uc in use_cases:
            lines.append(f"UseCase: {uc}")
        for rel in relationships:
            a = rel.get("actor")
            u = rel.get("use_case")
            if a and u:
                lines.append(f"Rel: {a} -> {u}")

        return "\n".join(lines)
    except Exception as e:
        return None  # Invalid JSON

def convert_and_save(split="train"):
    dataset = load_dataset("ReshmaUMLGraphMaster/Usecase-UML", split=split)
    processed = []

    for sample in dataset:
        prompt = sample["prompt"].strip()
        response = sample["response"].strip()
        dsl_output = json_to_dsl(response)
        if dsl_output:
            processed.append({
                "input_text": prompt,
                "output_text": dsl_output
            })

    os.makedirs("data/processed_dsl", exist_ok=True)
    output_file = f"data/processed_dsl/{split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for example in processed:
            f.write(json.dumps(example) + "\n")

    print(f"✅ Saved {len(processed)} valid DSL samples to {output_file}")

if __name__ == "__main__":
    convert_and_save("train")
