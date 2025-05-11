import torch
from model.transformer import Transformer
from preprocessing.tokenizer import Tokenizer
from diagram_generator.uml_drawer import draw_use_case
from interface.dsl_to_json import dsl_to_json

def generate(model, tokenizer, input_text, device, beam_width=3, max_len=64):
    model.eval()
    input_ids = tokenizer.encode(input_text)
    src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    sos_id = tokenizer.word2idx["<sos>"]
    eos_id = tokenizer.word2idx["<eos>"]

    sequences = [[ [sos_id], 0.0 ]]  # [tokens, score]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == eos_id:
                all_candidates.append((seq, score))
                continue

            tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            with torch.no_grad():
                output = model(src_tensor, tgt_tensor)
                probs = torch.log_softmax(output[:, -1, :], dim=-1)
                topk_probs, topk_ids = torch.topk(probs, beam_width)

            for i in range(beam_width):
                next_token = topk_ids[0, i].item()
                new_seq = seq + [next_token]
                new_score = score + topk_probs[0, i].item()
                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

    best_sequence = sequences[0][0]
    decoded = tokenizer.decode(best_sequence[1:], skip_special=True)

    # Clean duplicate lines
    lines = decoded.split("\n")
    clean_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))
    return "\n".join(clean_lines)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load("data/vocab/vocab.json")
    vocab_size = len(tokenizer.vocab)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4
    ).to(device)

    model.load_state_dict(torch.load("outputs/checkpoints/transformer.pth", map_location=device))

    story = input("\nInput Story:\n ")
    raw_output = generate(model, tokenizer, story, device=device)

    print("\n📝 DSL Output:\n", raw_output)

    try:
        diagram_data = dsl_to_json(raw_output)
        draw_use_case(diagram_data, output_path="uml_output")
    except Exception as e:
        print("❌ Could not draw diagram:", e)
