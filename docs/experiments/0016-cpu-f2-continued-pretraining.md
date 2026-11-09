# Experiment 0016: CPU F2 continued pre-training on a second FineWeb-Edu shard

**Date:** 2026-11-09
**Status:** Completed (F2 checkpoint)
**Hardware:** 2 CPU cores, 8 GB RAM, no GPU, FP32
**Parent:** F1, [Experiment 0012](0012-cpu-math-adaptation-finemath.md) (`f1-math-6m-step12500`)
**Weights:** [`checkpoints` branch, `f2-edu2-6m-step17500/`](https://github.com/FAHTHIZ/Artemus/tree/checkpoints/f2-edu2-6m-step17500) (sha256 `c4a791f9…d584`)

## Purpose

Experiments 0013–0014 showed that more SFT does not make a 6.45M model more capable; base-model quality is the bottleneck. F2 adds fresh, high-quality, deduplicated pre-training tokens.

## Data (`data/f2-mix`)

| Source | Document weight | Notes |
|---|---:|---|
| FineWeb-Edu sample-10BT shard 000, edu score ≥ 3.5 | 0.45 | same pool as F0 |
| FineWeb-Edu sample-10BT shard 007 (rev `87f09149`), edu score ≥ 3.5 | 0.45 | new: 87,565 accepted of 714k; 6 validation overlaps + 563 exact duplicates of shard-000 train removed → 86,996 docs |
| FineMath-4+ | 0.10 | replay to keep the F1 maths gains |

190k train docs / 900 validation docs, seed 1340, train/validation hash overlap 0. 207.4M train tokens (8k SentencePiece tokenizer).

## Optimisation

| Setting | Value |
|---|---|
| Steps | 12,500 → 17,500 (+5,000, batch 16 × 256 ≈ 20.5M tokens ≈ 0.1 epoch) |
| LR | 200 warmup steps to 8e-4, cosine schedule planned for 10,000 steps |
| Early anneal | at step 16,250, the run was re-targeted to 17,500: cosine from 1.94e-4 to 8e-5 over 1,250 steps |
| Seed | 2030 |
| Config | `scripts/cpu/f2.json` via the stage supervisor |

Reason for the early anneal: the sandbox suspends background processes while the session is idle, so the remaining 6k steps would have taken many hours of wall-clock time. Annealing produced a usable, fully decayed checkpoint now. F3 can continue from it on the same mix with a new warmup.

## Results

| Step | LR | Mixed validation loss |
|---:|---:|---:|
| 13,500 | ~7.5e-4 | 3.796 |
| 15,000 | ~6.9e-4 | 3.759 |
| 16,250 | ~6.5e-4 | 3.734 |
| 16,500 | 1.83e-4 | 3.691 |
| 17,000 | 1.19e-4 | 3.674 |
| 17,500 | 8.0e-5 | **3.664** |

Held-out bits/byte (`artemus-lm-eval`, 300 docs each, the same files as every earlier stage):

| Checkpoint | FineWeb-Edu | FineMath |
|---|---:|---:|
| F0 (10,000) | 1.346 | 1.887 |
| F1 (12,500) | 1.350 | 1.508 |
| **F2 (17,500)** | **1.321** | **1.507** |

## Samples (temperature 0.7, top-k 40)

> The French Revolution began in 1792, and its succession was to put the British into the country. …

> To solve the equation 2x + 6 = 10, we find that the equation 2x + 6 = 10 + 6 = 10 3, …

Fluent, topical English; facts and arithmetic are still wrong.

## Interpretation

- F2 is the best base model so far: −0.029 bpb on FineWeb-Edu and FineMath preserved. The anneal alone gave about half of the validation-loss improvement (3.734 → 3.664), consistent with WSD-style schedules.
- Next: S3 joint SFT from F2 (Experiment 0017), then F3 continued pre-training.
