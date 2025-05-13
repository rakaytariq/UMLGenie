import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from diagram_generator.uml_drawer import draw_use_case
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

# Load model
model_dir = "t5_trained"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# BLEU utility
def compute_bleu(reference, prediction):
    smoothie = SmoothingFunction().method4
    ref_tokens = reference.strip().split()
    pred_tokens = prediction.strip().split()
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    return round(score * 100, 2)

# DSL + UML Generation
def generate_dsl(story, reference_dsl=""):
    input_ids = tokenizer(story, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    dsl = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # UML generation
    diagram_path = None
    try:
        draw_use_case(dsl)
        diagram_path = "uml_output.png"
    except Exception as e:
        dsl += f"\n\n❌ UML render failed: {e}"

    # BLEU score
    bleu_score = compute_bleu(reference_dsl, dsl) if reference_dsl.strip() else "N/A"

    return dsl, bleu_score, diagram_path

# Gradio UI
iface = gr.Interface(
    fn=generate_dsl,
    inputs=[
        gr.Textbox(lines=4, label="User Story"),
        gr.Textbox(lines=4, label="Reference DSL (optional)")
    ],
    outputs=[
        gr.Textbox(label="Generated DSL"),
        gr.Textbox(label="BLEU Score (%)"),
        gr.Image(type="filepath", label="UML Diagram")
    ],
    title="UMLGenie: T5-Powered DSL & UML Generator",
    description="Enter a natural language story. Optionally add a reference DSL to evaluate BLEU score."
)

if __name__ == "__main__":
    iface.launch()
