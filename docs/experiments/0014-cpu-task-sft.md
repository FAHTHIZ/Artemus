# Experiment 0014 — CPU task SFT (OASST1 + GSM8K train + MBPP train)

**Date:** 2026-11-09
**Status:** Completed (S2 checkpoint)
**Hardware:** 2 CPU cores, 8 GB RAM, no GPU, FP32
**Parent:** S1, [Experiment 0013](0013-cpu-aya-instruction-sft.md) (`s1-aya-6m-step13300`)
**Weights:** [`checkpoints` branch, `s2-task-6m-step14800/`](https://github.com/FAHTHIZ/Artemus/tree/checkpoints/s2-task-6m-step14800) (sha256 `2f2532e3…`)

## Purpose

Extend the S1 chat model to conversational help (OASST1), step-by-step grade-school maths (GSM8K) and short Python functions (MBPP), using only human-written training splits.

## Data (`data/capability`, built by the repository capability pipeline)

| Source | Licence | Used for training | Held out |
|---|---|---|---|
| OASST1 reviewed English conversations | Apache-2.0 | conversation-tree grouped train split | grouped validation trees |
| GSM8K official **train** (rev `3101c7d5`) | MIT | 6,011 examples | hash bucket 0 = validation, bucket 1 = diagnostic eval; official test never downloaded |
| MBPP (rev `83f1e7b5`) | CC-BY-4.0 | task ids 601–974 (374) | 511–555 validation, 556–600 diagnostic eval; 11–510 protected test excluded |

## Optimisation

| Setting | Value |
|---|---|
| Objective | response-only causal LM loss |
| Steps | 13,300 → 14,800 (+1,500, batch 16 × 256) |
| LR | 50 warmup steps to 2e-4, cosine to 2e-5 |
| Seed | 2029 |
| Config | `scripts/cpu/s2.json` via `scripts/cpu/stage_supervisor.py s2` |

## Results

| Step | Capability validation loss |
|---:|---:|
| 13,500 | 3.252 |
| 13,800 | 3.180 |
| 14,100 | 3.139 |
| 14,400 | 3.108 |
| 14,800 | **3.102** (train 2.925) |

Held-out language modelling (300 docs each, `artemus-lm-eval`):

| Checkpoint | FineWeb-Edu bits/byte | FineMath bits/byte |
|---|---:|---:|
| F1 (base) | 1.350 | 1.508 |
| S1 (Aya SFT) | 1.479 | 1.692 |
| S2 (task SFT) | 1.410 | 1.632 |

Internal held-out capability diagnostic (`artemus-capability-eval`, greedy, 96 new tokens, not official benchmark scores):

| Checkpoint | Math exact match | Math answers parsed | Python syntax-valid |
|---|---:|---:|---:|
| F1 | 0/100 | 5/100 | 6/38 |
| S2 | 0/100 | 12/100 | 0/38 |

## Samples (temperature 0.7, top-k 40)

> **Q:** Tom has 3 apples and buys 5 more. How many apples does he have? **A:** The second one is 5 + 3 apples = <<5+3=2>>2 apples. …

> **Q:** What is the capital of France? **A:** The capital of France is a country located in the West Indies … `<|end|>`

## Interpretation

- S2 picked up the GSM8K calculator-annotation format and doubled the parseable-answer rate, but it cannot do the arithmetic itself. The Python outputs drift into maths prose, which is why syntax validity dropped.
- Task SFT recovered part of the general-text bits/byte lost in S1 (OASST1 is broader than Aya English).
- S2 answered the France question worse than S1 did. With this little capacity, newer SFT data overwrites earlier facts.
- Conclusion: at 6.45M parameters, more SFT does not make the model capable. What it needs is more pre-training tokens and parameters, which means the Kaggle GPU path (~24.4M params) and F2.

## Next

F2: 10,000 further pre-training steps from F1 on a fresh FineWeb-Edu shard (007, score ≥ 3.5, deduplicated against shard 000 and all validation sets) plus FineMath replay. Then repeat SFT on top of F2.
