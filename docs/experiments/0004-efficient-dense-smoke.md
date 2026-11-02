# Experiment 0004 — Efficient dense architecture smoke test

**Date:** 2026-11-02  
**Status:** Completed; 23 tests passed and 10-step CPU smoke training completed  
**Branch:** `research/efficient-dense-v2`  
**Workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41372580511

## Objective

Verify a backward-compatible architecture candidate with:
- Grouped-query attention (4 query heads, 2 key/value heads).
- Rotary position embeddings (RoPE), removing learned absolute position embeddings.
- Per-head query/key RMSNorm.
- SwiGLU feed-forward blocks.

The legacy full-MHA + learned-position + GELU configuration remains the default, preserving historical parameter names and shapes.

## Validation

- Unit and architecture tests: **23 passed**.
- Strict state-dict compatibility: a legacy configuration/checkpoint loads with no missing or unexpected keys.
- Reload equality: identical logits after strict save/load using legacy configuration.
- GQA forward pass, finite loss, backward gradients, and causal masking: passed.
- RoPE shape and per-vector norm preservation: passed.
- GQA head divisibility and even rotary-dimension validation: passed.

## CPU smoke model

| Setting | Value |
|---|---:|
| Parameters | 108,992 |
| Layers | 2 |
| Query heads | 4 |
| KV heads | 2 |
| Width | 64 |
| Context | 32 |
| Vocabulary | 257 |
| Batch | 4 |
| Steps | 10 |
| Device | CPU |
| Tokenizer | Byte fallback |
| Features | GQA + RoPE + QKNorm + SwiGLU |

Training loss moved from 5.5284 at step 1 to 5.1763 at step 10; validation loss moved from 5.5106 to 5.1579. This is only a tiny demonstration-corpus smoke test.

## Parameter comparison

At the same small width/depth/vocabulary/context:
- Original full-MHA + learned position embedding + GELU FFN: approximately **117,760** parameters.
- V2 GQA + RoPE + QKNorm + SwiGLU: **108,992** parameters.
- Net difference: approximately **7.4% fewer parameters** for this particular combined configuration.

This is a compound ablation. It does not isolate which component caused which change in quality. The attention projection test separately confirms that the GQA projection has fewer parameters than full MHA.

With 4 query heads and 2 KV heads, the K/V projection and a future compact KV cache can use half as many head slots as full MHA at the same head width. **The current inference implementation does not yet have a KV cache** and explicitly expands K/V heads before PyTorch SDPA, so no decode-speed or runtime-memory improvement has yet been demonstrated.

## Assessment

The candidate is mechanically sound and slightly smaller at this configuration. It is not yet proven to be more capable or faster. Do not promote it based on a 10-step smoke run.

Next compare the baseline and V2 with the same dataset, tokenizer, token budget, optimiser, seed, evaluation tasks, and compute. Report:
- held-out validation and task success;
- total and active parameters;
- peak memory and tokens/s;
- prefill/decode latency;
- actual KV-cache bytes once cache support exists.

## Reproducibility

- Workflow: https://github.com/FAHTHIZ/Artemus/actions/runs/41372580511
- Implementation: `src/artemis/model.py`
- Configuration: `src/artemis/config.py`
- CLI controls: `src/artemis/train.py`
- Compatibility and architecture tests: `tests/test_model.py`, `tests/test_model_compat.py`
