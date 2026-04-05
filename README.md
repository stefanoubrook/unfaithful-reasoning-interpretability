# Detecting Unfaithful Reasoning in Language Models

## Overview

This project investigates whether internal representations in language models can be used to detect unfaithful or misleading reasoning.

In particular, it focuses on cases where a model’s explanation does not accurately reflect the computation underlying its answer, and explores whether this discrepancy can be identified from hidden states.

## Motivation

Models may generate explanations that appear plausible while not reflecting their actual reasoning process.

Being able to detect such unfaithful reasoning from internal representations could improve evaluation methods and help identify potential failure modes in advanced systems.

## Current Approach

The initial focus is on building a simple pipeline using small transformer models:

1. Generate synthetic reasoning tasks (e.g. arithmetic problems)
2. Prompt the model to produce both answers and explanations
3. Compare outputs to ground truth to identify inconsistencies
4. Extract hidden states from the model during generation
5. Prepare data for probing internal representations

## Current Progress

- Implemented task generation for simple reasoning problems  
- Running GPT-2 to generate answers and explanations  
- Extracting hidden states for each example  
- Building dataset for analysis of reasoning behaviour  

## Next Steps

- Define more precise criteria for “unfaithful” reasoning  
- Train simple probes on hidden states to detect behavioural differences  
- Analyse which layers contain useful signals  
- Extend to more complex tasks and models  

## Notes

This is an initial exploratory project. The current setup uses simple tasks and approximate labels, with the aim of iterating toward more robust definitions and evaluations.
