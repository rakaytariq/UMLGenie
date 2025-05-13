from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5_trained")
model = T5ForConditionalGeneration.from_pretrained("t5_trained")

def generate_dsl(text):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        user_input = input("Enter a story:\n> ")
        if user_input.lower() in ["exit", "quit"]: break
        print("\n📝 DSL Output:\n", generate_dsl(user_input), "\n")
