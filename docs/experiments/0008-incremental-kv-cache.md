# Experiment 0008 — Incremental KV cache implementation and CPU benchmark

**Date:** 2026-11-02  
**Status:** Completed; 47 tests passed; cached/uncached outputs matched  
**Branch:** `optimization/incremental-kv-cache`  
**Architecture workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41375944019  
**Validation workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41375943846

## Objective

Turn the GQA cache-size calculation into an actual inference path. The implementation must preserve legacy state-dict compatibility and produce the same logits/tokens as uncached inference.

## Implementation

- Added `forward_with_cache` to attention, TransformerBlock, and ArtemisLM.
- Stores per-layer K/V tensors in compact `[batch, n_kv_head, context, head_dim]` shape.
- Applies RoPE using the absolute token position when continuing from cache.
- Uses learned absolute-position offsets when the legacy position embedding is enabled.
- Prefill uses normal causal SDPA.
- Cached continuation uses an explicit absolute-position mask. For non-square attention, this avoids the incorrect upper-left alignment that can result from using `is_causal=True` with a short query and long K/V sequence.
- Supports multi-token cached chunks, not only one-token decode.
- Cached generation is now the default; `--no-cache` preserves the full-context reference path.
- Re-prefills from the latest context window when generation rolls beyond `block_size`, matching the old sliding-window behaviour.
- Added `artemus-bench-kv-cache` to compare latency, generated tokens/s, output equality, and actual cache tensor payload bytes.

No model weights or state-dict parameter names were changed.

## Correctness verification

**47 tests passed** in the architecture CI workflow, including:
- Full-sequence forward logits versus token-by-token cached logits for both legacy MHA/learned positions and compact GQA/RoPE/QKNorm/SwiGLU/factorized embeddings.
- Full-sequence logits versus prefill plus multi-token cached continuation chunks.
- Cache shape checks proving GQA stores `n_kv_head`, not expanded query-head count.
- Cached versus uncached generation output equality across context-window rollover.
- Cached generation restores the model's original training/eval mode.
- Existing training, tokenizer, checkpoint, model, and profile tests.

The separate validation workflow also passed all unit/data-integrity tests and Python syntax checks.

## Benchmark configuration

This first systems measurement used a tiny random-init smoke model, not trained weights:

| Setting | Value |
|---|---:|
| Parameters | 113,024 |
| Vocabulary | 257 (byte tokenizer size) |
| Layers | 2 |
| Query heads | 4 |
| KV heads | 2 |
| Width | 64 |
| Factorised embedding width | 32 |
| Position | RoPE |
| QK norm | Enabled |
| Feed-forward | SwiGLU |
| Device | GitHub-hosted CPU |
| CPU threads | 2 |
| Prompt | 32 tokens |
| Generated | 24 tokens |
| Warmup / timed repeats | 1 / 3 |
| Sampling | top-k 1, identical deterministic output |

## Measured result

| Measurement | Cached | Uncached |
|---|---:|---:|
| Median elapsed time | 27.887 ms | 37.181 ms |
| Generated tokens per second | 860.6 | 645.5 |
| Generated output equality | `true` | `true` |

Uncached/cached elapsed-time ratio: **1.333×** in this single CI CPU configuration.

Persistent K/V payload at the prompt context:
- GQA cache actually stored: **16,384 bytes**.
- Full-MHA cache estimate at the same layer count, context, head dimension, and dtype: **32,768 bytes**.
- Payload reduction: **50%**.

## Interpretation and limits

This is the first measured decode improvement from caching on the tested tiny CPU setup. It is not a general speedup guarantee.

- The benchmark includes prompt prefill and token sampling in the total generation time; it does not separately isolate prefill and per-token decode latency.
- The model is random-initialized and only 113K parameters. Results will change materially with model size, trained weights, context length, batch, device, and kernel implementation.
- K/V cache tensors are concatenated as tokens arrive. A preallocated/static cache could reduce repeated copy overhead.
- GQA K/V heads are temporarily expanded before PyTorch SDPA in this compatibility implementation. The persistent cache remains compact, but a native GQA kernel could reduce temporary allocations.
- The payload estimate excludes model weights, activations, temporary expanded tensors, allocator fragmentation, and runtime buffers.
- This CI run uses CPU only; no CUDA/MPS comparison has been performed.

## Next engineering steps

1. Add a preallocated/static KV cache to reduce per-token concatenation/copy overhead.
2. Benchmark prefill and decode separately at several context lengths and batch sizes.
3. Compare compact GQA versus full MHA on a larger model, and run on the actual target accelerator when available.
4. Measure peak memory, throughput distributions, and energy/cost where accessible.
5. Keep the cache only if correctness remains exact within numerical tolerance and real target-hardware performance justifies it.
