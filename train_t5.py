from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

# Load and convert to Hugging Face Dataset
df = pd.read_csv("t5_dsl_dataset.csv").dropna()
dataset = Dataset.from_pandas(df)

# Initialize tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization
def preprocess(example):
    input = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=128)
    output = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=128)
    input["labels"] = output["input_ids"]
    return input

tokenized = dataset.map(preprocess, remove_columns=["input_text", "target_text"])

# Training configuration
args = TrainingArguments(
    output_dir="./t5_output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# Train and save model
trainer.train()
model.save_pretrained("t5_trained")
tokenizer.save_pretrained("t5_trained")
print("✅ Model trained and saved to t5_trained/")
