"""Generate simple arithmetic reasoning tasks."""

import json
import random
import argparse
from pathlib import Path


def generate_arithmetic_problems(n: int, seed: int = 11) -> list[dict]:
    """Generate n arithmetic problems (addition and subtraction)."""
    random.seed(seed)
    problems = []

    for i in range(n):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(["+", "-"])

        if op == "+":
            answer = a + b
        else:
            answer = a - b

        question = f"What is {a} {op} {b}?"
        problems.append({
            "id": i,
            "question": question,
            "ground_truth": answer,
            "operands": [a, b],
            "operator": op,
        })

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=str, default="data/raw/arithmetic.jsonl")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    problems = generate_arithmetic_problems(args.n, args.seed)

    with open(args.output, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"Generated {len(problems)} problems -> {args.output}")


if __name__ == "__main__":
    main()
