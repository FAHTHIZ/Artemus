# Efficient Dense Architecture V2 — Research Notes

**Status:** Experimental implementation; not selected as the production architecture.  
**Branch:** `research/efficient-dense-v2`

## Motivation

The first dense baseline used learned absolute position embeddings, full multi-head attention (MHA), and a GELU feed-forward block. It was easy to validate but did not prioritise inference-memory efficiency.

This branch adds optional architecture switches while preserving the original configuration as the default. Historical checkpoints with no new configuration fields should retain the same module names and tensor shapes.

## Changes

### Grouped-query attention (GQA)

When `n_kv_head < n_head`, Artemis projects queries to all query heads but projects keys and values to fewer KV heads. Each KV head is shared by a group of query heads.

For a model width (d), query-head count (H_q), and KV-head count (H_{kv}):

- Query projection width remains (d).
- Key and value projection widths become (H_{kv}d/H_q) each.
- KV cache storage scales approximately with (H_{kv}/H_q) relative to full MHA for the same layers, head width, sequence length, and batch size.

The current implementation expands KV heads with `repeat_interleave` before PyTorch SDPA for broad API compatibility. This reduces projected KV dimensions and is a correctness baseline; a future kernel that consumes grouped KV heads directly may avoid materialising the expanded view.

### Rotary position embeddings (RoPE)

When enabled, RoPE replaces learned absolute position embeddings and rotates Q/K channel pairs according to position. It removes the learned position-embedding matrix and supports position extrapolation experiments, but this implementation still uses a fixed configured context window and resets positions when generation crops to its most recent context window.

### QK normalization

Optional RMSNorm is applied per query/key head before rotary transforms. It adds a small number of learned scale parameters and is an explicit stability ablation, not an assumed quality gain.

### SwiGLU

Optional SwiGLU uses parallel gate/up projections and a down projection with hidden width chosen to keep feed-forward parameter count near the original two-matrix (4d)-width GELU block. Parameter parity is approximate after rounding hidden size to a multiple of eight.

## Backward compatibility

The default config keeps:
- Full MHA with the historical `qkv` projection name and shape.
- Learned position embedding with the historical `position_embedding` state key.
- The original GELU `net` feed-forward module and state keys.
- No QK norm.

Older serialized config dictionaries omit the new fields; dataclass defaults restore the original architecture. The compatibility test checks strict state-dict loading and exact output equality after reload.

## Research CI

The branch workflow:
- Runs the full unit suite.
- Trains a 10-step CPU smoke model using GQA, RoPE, QK norm, and SwiGLU.
- Saves a disposable local checkpoint only as a CI output; it is not promoted.

## Evaluation required before adoption

Compare old dense and V2 under the same seed, data, token budget, batch size, optimizer, and evaluation tasks.

Record:
- Total parameters and trainable parameters.
- KV cache bytes at fixed batch/context/dtype.
- Training tokens/s and peak memory.
- Prefill and decode tokens/s at multiple contexts.
- Validation loss and task success on held-out code/math/conversation tasks.
- Numerical stability and failure rates.

The implementation's parameter reduction is not proof of improved quality or end-to-end speed. CPU GQA currently expands KV heads before attention; its true performance benefit depends on kernels and hardware. Adopt only if a measured matched-budget comparison is positive.
