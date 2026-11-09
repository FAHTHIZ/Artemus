# Experiment 0013 — CPU English instruction SFT on the Aya Dataset

**Date:** 2026-11-09
**Status:** Completed (S1 checkpoint)
**Hardware:** 2 CPU cores, 8 GB RAM, no GPU, FP32
**Parent:** F1, [Experiment 0012](0012-cpu-math-adaptation-finemath.md) (`f1-math-6m-step12500`, sha256 `dcf408be…c9cb`)
**Weights:** [`checkpoints` branch, `s1-aya-6m-step13300/`](https://github.com/FAHTHIZ/Artemus/tree/checkpoints/s1-aya-6m-step13300)

## Purpose

Teach the base model the `<|prompter|>` / `<|assistant|>` / `<|end|>` turn format and short answer style using human-written instructions only (no model-generated data).

## Data

| Item | Value |
|---|---|
| Source | `CohereLabs/aya_dataset`, revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` |
| Licence | Apache-2.0 |
| Filter | `language_code == eng` only (repository policy: English only) plus standard PII/length/dedup gates |
| Split | 3,775 train / 83 validation conversations |

## Optimisation

| Setting | Value |
|---|---|
| Objective | response-only causal LM loss (prompt tokens masked), `--allow-objective-change` |
| Steps | 12,500 → 13,300 (+800, batch 16 × 256, ≈3.4 epochs) |
| LR | 50 warmup steps to 3e-4, cosine to 3e-5 |
| Seed | 2028 |
| Config | `scripts/cpu/s1.json` via `scripts/cpu/stage_supervisor.py s1` |

## Results

| Step | Train loss | Aya validation loss |
|---:|---:|---:|
| 12,600 | 3.263 | 3.496 |
| 12,800 | 2.858 | 3.380 |
| 13,000 | 2.632 | 3.337 |
| 13,300 | 2.515 | **3.314** |

Validation kept improving but the train/validation gap (0.8 nats) shows the 3.8k-example set is being memorised; more epochs would overfit.

## Samples (temperature 0.7, top-k 40)

> **Q:** What is the capital of France? **A:** The capital of France is Paris. `<|end|>`

> **Q:** What is photosynthesis? **A:** The photosynthesis process is a complex process that allows plants to convert sunlight into glucose, … Water is crucial for digestion, digestion, and digestion. `<|end|>`

> **Q:** Give me three tips for staying healthy. **A:** Your healthcare provider will probably be at risk for severe jaundice… `<|end|>`

## Interpretation

- Format learning succeeded: the model answers in the assistant turn and terminates with `<|end|>`.
- Content is mostly wrong beyond the simplest facts — the limit of a 6.45M-parameter base model, not the SFT data.
- English-only Aya is small; for the GPU run, combine it with OASST1 English and task SFT rather than adding epochs.

## Next

S2 task SFT (GSM8K / MBPP train splits + OASST1) and held-out capability evaluation.
