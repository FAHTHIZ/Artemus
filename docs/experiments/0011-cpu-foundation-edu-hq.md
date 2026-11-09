# Experiment 0011 — CPU foundation pretraining on FineWeb-Edu score ≥ 3.5

**Date:** 2026-11-09
**Status:** Completed (F0 foundation checkpoint)
**Hardware:** 2 CPU cores, 8 GB RAM, no GPU, FP32
**Weights:** GitHub Release [`f0-edu-6m-step10000`](https://github.com/FAHTHIZ/Artemus/releases/tag/f0-edu-6m-step10000) (not committed to Git)

## Purpose

Train the first ArtemisLM foundation checkpoint from random initialization on the cleanest English slice of the repository's already-approved source, and establish a tokenizer-independent held-out baseline for later runs.

## Data

| Item | Value |
|---|---|
| Source | `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, shard `sample/10BT/000_00000.parquet` |
| Licence | ODC-By 1.0; Common Crawl terms apply |
| Ingestion | `artemus-prepare-parquet` (bounded row batches; same gates as `artemus-prepare-hf`) |
| Language gate | English label, language score ≥ 0.70 |
| Quality gate | FineWeb-Edu classifier `score` ≥ 3.5 (new `--min-quality-score`) |
| Rows rejected for quality score | 627,247 (only ~12% of rows pass) |
| Rejected: PII/secret patterns / language / length / duplicates | 6,063 / 2,423 / 789 / 208 |
| Accepted | 88,355 train + 915 validation documents (377.9M characters) |
| Tokens (train / validation) | 94,045,762 / 908,182 |

The streaming `datasets` path was killed twice on this 8 GB machine (at ~16k and ~22k accepted documents) without writing a manifest. Reading a pinned Parquet shard in bounded batches avoided this; the GPU bundle builder now uses the same path for FineWeb-Edu.

## Tokenizer

SentencePiece BPE, 8,192 pieces, byte fallback, `split_digits=True`, chat markers reserved. Trained on a 60M-character random sample of the training split only. Held-out compression: 3.99 characters per token.

## Model and optimisation

| Setting | Value |
|---|---|
| Architecture | 6 layers, width 256, 4 query / 2 KV heads (GQA), RoPE, QK-norm, SwiGLU, tied embeddings |
| Parameters | **6,451,200** |
| Context | 256 tokens |
| Batch | 16 sequences (4,096 tokens/step) |
| Steps | 10,000 (≈41M token presentations, ≈0.44 epochs; ≈6.3 tokens/parameter) |
| Optimiser | AdamW, peak LR 2e-3, 400 warmup steps, cosine to 2e-4, grad-clip 1.0 |
| Wall-clock | ≈3.6 h at ≈1.1–1.3 s/step |
| Supervisor | `scripts/cpu/f0_supervisor.py` (auto-resume on the same global cosine curve; no restarts were needed) |

## Results

Validation loss (20 eval batches): 8.87 (step 1) → 5.47 (500) → 4.42 (2,000) → 3.99 (5,000) → 3.80 (8,000) → **3.749 (10,000)**. Train and validation loss stayed within 0.02 throughout; no overfitting.

Held-out `artemus-lm-eval` (300 documents each, full documents in 256-token windows):

| Held-out set | Token loss | Perplexity | Bits per byte |
|---|---:|---:|---:|
| FineWeb-Edu ≥ 3.5 validation | 3.767 | 43.2 | **1.346** |
| FineMath-4+ validation (unseen domain) | 3.758 | 42.9 | **1.887** |

## Samples (temperature 0.7, top-k 40)

> **Photosynthesis is the process by which** the energy of the plant can penetrate a cell membrane into a certain cell. Therefore, the energy of photosynthesis is a process called photosynthesis…

> **The French Revolution began in** the 17th century when the French Revolution began with the French Revolution. In 1767, the French Revolution had been defeated…

## Interpretation

- The model produces grammatical, on-topic English with educational register, but repeats itself and its facts are wrong. This is the expected capability of a ~6M-parameter model trained on ~41M tokens; it is a foundation checkpoint, not an assistant.
- The run was still improving at the end (loss −0.10 over the last 2,000 steps at low LR). More tokens and more parameters are the main levers; both need a GPU.
- Bits per byte gives the first tokenizer-independent baseline. Future runs (different vocabularies or sizes) should report BPB on the same held-out files.

## Next

1. F1: math domain adaptation (40% FineMath-4+ / 60% FineWeb-Edu replay) from this checkpoint.
2. S1/S2: response-only English instruction SFT (Aya `eng`, OASST1) and task SFT (GSM8K/MBPP train splits).
3. Run the upgraded ~24.4M-parameter GPU configuration on Kaggle.
