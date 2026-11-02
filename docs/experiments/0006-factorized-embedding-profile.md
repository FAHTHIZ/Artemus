# Experiment 0006 — Factorized embedding parameter profile

**Date:** 2026-11-02  
**Status:** Completed; 33 tests and factorized training smoke passed  
**Branch:** `research/efficient-dense-v2`  
**Workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41373883271

## Objective

Measure the parameter impact of reducing token embedding width while retaining the Transformer model width and tied input/output vocabulary matrix.

The factorized architecture projects token embeddings from (d_e) to (d) before the Transformer, then projects final hidden states back from (d) to (d_e) before the tied vocabulary output.

## Profiled configuration

| Setting | Value |
|---|---:|
| Vocabulary | 4,096 |
| Layers | 4 |
| Query heads | 4 |
| KV heads | 2 |
| Model width (d) | 128 |
| Factorized embedding width (d_e) | 64 |
| Context | 256 |
| Position | RoPE |
| QK normalization | Enabled |
| Feed-forward | SwiGLU |

## Parameter comparison

| Component | Non-factorized candidate | Factorized candidate |
|---|---:|---:|
| Tied token/output matrix | 524,288 | 262,144 |
| Input/output projections | 0 | 16,384 |
| Position embeddings | 0 | 0 |
| Attention, all layers | 196,864 | 196,864 |
| Feed-forward, all layers | 528,384 | 528,384 |
| Block norms | 1,024 | 1,024 |
| Final norm | 128 | 128 |
| **Total** | **1,250,688** | **1,004,928** |

Factorization saves **245,760 parameters**, a **19.65% reduction** relative to the non-factorized GQA/RoPE/QKNorm/SwiGLU candidate.

Against the old full-MHA + learned position embedding + GELU baseline at the same vocabulary, width, layer count, and context (analytical count: 1,347,200), the factorized compact candidate is 342,272 parameters smaller, approximately **25.4% fewer**. This is a count comparison, not a matched-quality result.

## Validation

- Total parameter accounting excludes double-counting the tied output embedding.
- Factorized input and output projections have expected shapes.
- Forward loss and backward gradients are finite.
- Legacy configs still instantiate identity projections and preserve original token/output matrix shapes.
- Full unit and architecture suite: **33 passed**.
- 10-step CPU factorized GQA/RoPE/QKNorm/SwiGLU smoke training completed.

The short smoke loss is not a meaningful quality benchmark and is not compared with other training objectives.

## Limits and next test

Factorized embeddings reduce parameter storage but add two projections and may constrain the representational capacity of rare tokens, code identifiers, mathematical notation, Bengali, and Banglish.

The next comparison must train baseline and factorized candidates on identical tokenizer/data/seed/token budgets, then evaluate held-out language, coding, mathematics, and multilingual tasks. Retain the smaller embedding only if quality survives.

The current code still does not implement incremental KV caching. The theoretical GQA KV-cache estimate remains separate from measured inference memory or speed.
