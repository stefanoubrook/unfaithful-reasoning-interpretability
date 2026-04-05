"""Prompt GPT-2 to answer arithmetic questions with explanations."""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Answer the question and briefly explain your reasoning.\n"
        "Answer:"
    )


def generate_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 60
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/arithmetic.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/model_outputs.jsonl")
    parser.add_argument("--model_name", type=str, default="gpt2")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    problems = []
    with open(args.input) as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"Running inference on {len(problems)} problems...")
    with open(args.output, "w") as f:
        for i, problem in enumerate(problems):
            prompt = build_prompt(problem["question"])
            response = generate_response(model, tokenizer, prompt)

            record = {
                **problem,
                "prompt": prompt,
                "model_response": response,
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(problems)}")

    print(f"Saved outputs -> {args.output}")


if __name__ == "__main__":
    main()
