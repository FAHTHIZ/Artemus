# Experiment 0012 — CPU math domain adaptation on FineMath-4+

**Date:** 2026-11-09
**Status:** Completed (F1 checkpoint)
**Hardware:** 2 CPU cores, 8 GB RAM, no GPU, FP32
**Parent:** F0, [Experiment 0011](0011-cpu-foundation-edu-hq.md) (`f0-edu-6m-step10000`, sha256 `10f84469…b9a`)
**Weights:** [`checkpoints` branch, `f1-math-6m-step12500/`](https://github.com/FAHTHIZ/Artemus/tree/checkpoints/f1-math-6m-step12500) (kept off `main`)

## Purpose

Add mathematical text to the foundation model without losing its general English, using a replay mix rather than math-only data.

## Data

| Item | Value |
|---|---|
| Math source | `HuggingFaceTB/finemath`, config `finemath-4plus`, revision `e92b25a616738fe95dc186b64dfb19f9c8525594`, shard `train-00011-of-00064` |
| Licence | ODC-By 1.0; Common Crawl terms apply |
| Math gates | English, language score ≥ 0.70, length 300–30,000 chars, PII/secret regex, repetition and exact-dedup filters |
| Math accepted | 39,564 train + 436 validation documents |
| Replay source | F0's FineWeb-Edu score ≥ 3.5 train split (same files as Experiment 0011) |
| Mix | `deterministic_weighted_document_sampling`, seed 1338: 9,600 FineMath (38.7M chars) + 14,400 FineWeb-Edu (60.7M chars) = 24,000 train docs; 600 validation docs |

Held-out FineMath and FineWeb-Edu validation files are disjoint from all training data.

## Optimisation

| Setting | Value |
|---|---|
| Model | unchanged (6,451,200 params, 6L/256, GQA 4/2, RoPE, QK-norm, SwiGLU, 256 ctx) |
| Steps | 10,000 → 12,500 (+2,500 steps, ≈10.2M tokens, batch 16 × 256) |
| LR | 100 warmup steps to 6e-4, cosine to 6e-5 (resumed optimizer state, `--override-resume-learning-rate`) |
| Seed | 2027 |
| Supervisor | `scripts/cpu/stage_supervisor.py f1` with `scripts/cpu/f1.json` |
| Wall-clock | ≈50 min |

## Results

Mixed validation loss at step 12,500: **3.403** (train 3.489).

Held-out `artemus-lm-eval` (300 documents each, same files as Experiment 0011):

| Held-out set | F0 bits/byte | F1 bits/byte | Change |
|---|---:|---:|---:|
| FineMath-4+ validation | 1.887 | **1.508** | −20.1% |
| FineWeb-Edu ≥ 3.5 validation | 1.346 | 1.350 | +0.3% |

FineMath token perplexity fell from 42.9 to 20.2. General-text forgetting is negligible thanks to the 60% replay.

## Samples (temperature 0.7, top-k 40)

> **To solve the equation 2x + 3 = 11, we** can find the equation 2x + 2y = 12 – 3y = 26 + 4y = 25 – 3y …

> **In mathematics, a prime number** is a prime number. As a result, the prime number is a prime number…

## Interpretation

- The model has learned the surface form of mathematical writing (equations, LaTeX, worked-solution phrasing) but not arithmetic or reasoning; at 6.45M parameters and ~51M total tokens this is expected.
- A 40/60 domain/replay mix is a safe default for future domain stages: large in-domain gain, ~0 out-of-domain loss.

## Next

S1 (English Aya instruction SFT, response-only) and S2 (GSM8K/MBPP/OASST1 task SFT) from this checkpoint, then held-out capability evaluation.
