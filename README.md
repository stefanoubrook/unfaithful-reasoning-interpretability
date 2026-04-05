# Detecting Unfaithful Reasoning in Language Models

## Overview

This project investigates whether internal representations in language models can be used to detect unfaithful or misleading reasoning.

In particular, it focuses on cases where a model's explanation does not accurately reflect the computation underlying its answer, and explores whether this discrepancy can be identified from hidden states.

## Motivation

Models may generate explanations that appear plausible while not reflecting their actual reasoning process.

Being able to detect such unfaithful reasoning from internal representations could improve evaluation methods and help identify potential failure modes in advanced systems.

## Pipeline

```
generate_data.py → run_model.py → score_outputs.py → extract_activations.py
```

1. **Generate data** — synthetic arithmetic problems with ground truth answers
2. **Run model** — prompt GPT-2 to answer with explanations
3. **Score outputs** — extract numeric answers, compare to ground truth, label correctness
4. **Extract activations** — save final-layer hidden states for each example

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline end-to-end:

```bash
python src/generate_data.py --n 100
python src/run_model.py
python src/score_outputs.py
python src/extract_activations.py
```

Outputs:
- `data/raw/arithmetic.jsonl` — generated problems
- `data/processed/model_outputs.jsonl` — model responses
- `data/processed/scored_outputs.jsonl` — scored with correctness labels
- `results/activations.npz` — hidden states (n_examples, hidden_dim) + labels

## Project Structure

```
detecting-unfaithful-reasoning/
├── src/
│   ├── generate_data.py
│   ├── run_model.py
│   ├── score_outputs.py
│   └── extract_activations.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── results/
├── requirements.txt
└── .gitignore
```

## Next Steps

- Train simple probes on hidden states to detect correctness from activations
- Analyse which layers contain the most signal
- Define more precise criteria for "unfaithful" reasoning
- Extend to more complex tasks and models

## Notes

This is an initial exploratory project. The current setup uses simple tasks and approximate labels, with the aim of iterating toward more robust definitions and evaluations.
