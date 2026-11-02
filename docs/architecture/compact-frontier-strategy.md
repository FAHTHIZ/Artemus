# Compact Frontier-Capability Strategy

**Invariant:** Foundation weights start from random initialization inside Artemis.  
**Objective:** Maximise useful capability per stored parameter, active parameter, FLOP, byte of memory, watt, and unit of training compute.

## Capability target

The long-term ambition is frontier-class ability across reasoning, coding, mathematics, multilingual communication, and knowledge work with a smaller deployed model. This is an engineering target, not a guaranteed outcome. The current one-million-parameter runs are pipeline experiments and their generated answers are not yet coherent.

## Dense architecture hypotheses

Retain a reproducible dense baseline and change one variable per experiment. Candidates:
- GQA to reduce KV-head projections and eventual KV-cache storage.
- RoPE to replace learned absolute position embeddings.
- QK normalization as an explicit stability ablation.
- SwiGLU with a matched approximate feed-forward parameter budget.
- Factorized token embeddings/output projection.
- Weight-shared depth or adaptive compute, only after dense baselines are stable.

## Factorized token embeddings

With a large vocabulary, a tied token embedding/output matrix may be a substantial portion of the model. Factorize it with embedding width (d_e) smaller than model width (d). Project token embeddings from (d_e) to (d) before the Transformer; project final hidden states back to (d_e) before the tied vocabulary output.

For vocabulary (V), the token table changes from (Vd) parameters to (Vd_e), while the two projection matrices add approximately (2dd_e). It reduces parameter count when:

[
Vd > Vd_e + 2dd_e
]

Example at (V=4096), (d=128), (d_e=64):
- Original tied table: 524,288 parameters.
- Factorized table: 262,144 parameters.
- Two projections: 16,384 parameters.
- Net saving: 245,760 parameters.

This saves parameters but adds projection compute and may bottleneck information flow. Test quality on general text, rare code identifiers, mathematical notation, Bengali, and Banglish. Do not assume the lowest embedding width wins.

## Sparse models and active parameters

MoE may increase total stored capacity while activating a small fraction per token. Always report total and active parameters separately. Routing, communication, expert balancing, memory, and inference overhead matter. Do not call a 1B-total/200M-active model a 200M-parameter model.

## Measurement

For every matched-budget candidate record:
- Total/active parameters and component breakdown.
- Training tokens by source and mixture.
- Hardware, wall-clock, estimated FLOPs, memory, training tokens/s.
- Prefill/decode tokens/s, inference memory, and cache bytes.
- Held-out quality by capability, not only language-model loss.
- Multi-seed uncertainty and regressions where practical.

The current GQA branch expands KV heads before SDPA and has no incremental KV cache. Its compact cache calculation is theoretical until a real cache implementation and benchmark exist.

## Staged training

1. Tokenizer and provenance audit.
2. Random-init foundation pretraining on a broad, licensed, deduplicated corpus.
3. Instruction tuning on reviewed prompt/response data.
4. Verifiable code/math tasks and sandboxed execution feedback.
5. Outcome optimisation where rewards are checkable.
6. High-quality data anneal and held-out evaluation.

OASST1, GSM8K, and MBPP form an initial small task mixture, not a foundation-scale corpus. FineWeb is a future pretraining candidate, but its scale, legal terms, residual PII and content quality require careful handling.

## Promotion

A compact candidate wins only when it preserves or improves held-out task quality under comparable compute and inference constraints. No architecture is selected by parameter count alone, and no frontier parity claim is allowed without independent, comparable evidence.
