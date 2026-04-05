"""Extract hidden states from GPT-2 for each example."""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_hidden_states(
    model, tokenizer, text: str, layer: int = -1
) -> np.ndarray:
    """Get the hidden state at the final token from the specified layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers + 1) tensors, each (1, seq_len, hidden_dim)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    final_token = hidden[0, -1, :]  # (hidden_dim,)
    return final_token.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/scored_outputs.jsonl")
    parser.add_argument("--output", type=str, default="results/activations.npz")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (-1 = last)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Extracting activations for {len(records)} examples...")
    all_activations = []
    labels = []

    for i, record in enumerate(records):
        # Use the full prompt + response as input
        full_text = record["prompt"] + " " + record["model_response"]
        activation = get_hidden_states(model, tokenizer, full_text, layer=args.layer)
        all_activations.append(activation)
        labels.append(record["correct"])

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(records)}")

    activations_array = np.stack(all_activations)  # (n_examples, hidden_dim)
    labels_array = np.array(labels, dtype=bool)

    np.savez(
        args.output,
        activations=activations_array,
        labels=labels_array,
    )
    print(f"Saved activations {activations_array.shape} -> {args.output}")


if __name__ == "__main__":
    main()
