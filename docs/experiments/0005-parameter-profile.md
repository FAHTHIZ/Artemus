# Experiment 0005 — Parameter and KV-cache profile

**Date:** 2026-11-02  
**Status:** Completed; profiler and accounting tests passed  
**Branch:** `research/efficient-dense-v2`  
**Workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41373553170

## Configuration profiled

| Setting | Value |
|---|---:|
| Vocabulary | 4,096 |
| Layers | 4 |
| Query heads | 4 |
| KV heads | 2 |
| Model width | 128 |
| Head dimension | 32 |
| Context window | 256 |
| Batch size | 1 |
| KV-cache dtype estimate | 2 bytes (FP16/BF16) |
| Position encoding | RoPE |
| Normalisation | RMSNorm + QK norm |
| Feed-forward | SwiGLU |

## Parameter breakdown

The profiler reported:

| Component | Parameters |
|---|---:|
| Tied token embedding/output matrix | 524,288 |
| Attention across all layers | 196,864 |
| Feed-forward across all layers | 528,384 |
| Block RMSNorms | 1,024 |
| Final RMSNorm | 128 |
| Learned position embedding | 0 |
| **Total** | **1,250,688** |

The input token embedding and output projection share one weight matrix; the count includes it once.

For the same vocabulary, layer count, model width, and context, the legacy full-MHA + learned position embedding + GELU baseline has an analytically calculated **1,347,200 parameters**. The compact variant is 96,512 parameters smaller, approximately **7.17% fewer**. This is a combined change; the parameter differences are not a controlled measure of quality impact.

## KV-cache arithmetic

For a conventional decoder cache holding key and value tensors:

`batch × layers × context × 2(K,V) × KVHeads × headDim × bytes`

At batch 1, 4 layers, 256 tokens, head dimension 32, and 2 bytes per value:

- GQA with 2 KV heads: **262,144 bytes** (256 KiB).
- Full MHA with 4 KV heads: **524,288 bytes** (512 KiB).
- Theoretical reduction: **50%**.

This is a theoretical compact cache layout. Current Artemis generation recomputes full context at every token and does not yet implement an incremental KV cache. The current GQA attention implementation expands KV heads before PyTorch SDPA. No measured latency or peak-memory benefit is claimed.

## Tests and limitations

- Profiler tests verify the unique tied-parameter count.
- GQA cache arithmetic is checked against full MHA.
- Context length, dtype width, and architecture options are validated.
- The architecture CI also validates GQA/RoPE/QKNorm/SwiGLU forward/backward and causal masking.

This experiment is arithmetic and implementation validation—not model-quality evidence. A matched-budget training run with held-out English, coding and mathematics tasks is required before choosing the architecture.
