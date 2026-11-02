# Experiment 0009 — Compact model size ladder profile

**Date:** 2026-11-02  
**Status:** Completed; profiler run passed  
**Workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41376170869  
**Branch:** `main`

## Purpose

Turn the roadmap's 10M/30M/100M research ladder into concrete, instantiable model configurations. This experiment only constructs each model and counts unique parameters. It does not train them or claim any capability.

All three use:
- Random-init dense Transformer weights.
- RoPE, QK normalization, SwiGLU, and factorized token embeddings.
- Grouped-query attention.
- Tied token embedding/output matrix.
- Model width preserved separately from factorized embedding width.

## Exact profile results

| Candidate | Layers | Width | Query/KV heads | Vocabulary | Embedding width | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Small research model | 6 | 320 | 5 / 1 | 8,192 | 192 | **8,105,792** |
| Mid research model | 8 | 512 | 8 / 2 | 16,384 | 256 | **26,519,040** |
| Larger research model | 12 | 768 | 12 / 4 | 32,768 | 384 | **88,690,944** |

All profiles passed on the GitHub-hosted CPU runner. The 100M-class profile was instantiated successfully without training.

## Interpretation

These are useful next rungs above the current approximately 1M-parameter pipeline model. They are still small research models, not frontier-capable models by parameter count alone.

The next serious experiment must compare at least two sizes with:
- The same source-aware training mixture and tokenizer protocol.
- A controlled number of unique training tokens and total token presentations.
- Comparable optimizer schedules and training FLOPs.
- Held-out language modelling and task evaluations.
- Memory, throughput, and inference-latency measurements.

The 100M-class candidate is a plausible first accelerator-backed scale experiment once the training pipeline supports packed/memory-mapped corpora and an affordable GPU environment is available. Free CPU runners can validate the data pipeline and architecture, not train a frontier flagship.
