"""Score model outputs by extracting numeric answers and comparing to ground truth."""

import json
import re
import argparse
from pathlib import Path


def extract_number(text: str) -> int | None:
    """Extract the first integer (possibly negative) from text."""
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group())
    return None


def score_record(record: dict) -> dict:
    """Add extracted_answer and correct fields to a record."""
    extracted = extract_number(record["model_response"])
    correct = extracted == record["ground_truth"] if extracted is not None else False

    return {
        **record,
        "extracted_answer": extracted,
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/model_outputs.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/scored_outputs.jsonl")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    scored = [score_record(r) for r in records]

    with open(args.output, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    n_correct = sum(1 for r in scored if r["correct"])
    n_extracted = sum(1 for r in scored if r["extracted_answer"] is not None)
    print(f"Scored {len(scored)} examples")
    print(f"  Answer extracted: {n_extracted}/{len(scored)}")
    print(f"  Correct: {n_correct}/{len(scored)} ({100 * n_correct / len(scored):.1f}%)")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
