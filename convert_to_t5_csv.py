# convert_to_t5_csv.py

import json
import pandas as pd

with open("data/processed_dsl/train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

rows = []
for item in data:
    input_text = item.get("input_text", "").strip()
    target_text = item.get("output_text", "").strip()
    if input_text and target_text:
        rows.append({"input_text": input_text, "target_text": target_text})

df = pd.DataFrame(rows)
df.to_csv("t5_dsl_dataset.csv", index=False)
print("✅ Converted to t5_dsl_dataset.csv")
