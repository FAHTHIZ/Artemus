# Artemis — Model Design and Flagship Research Plan

**Status:** Active research program; the production-scale dimensions are deliberately not fixed until matched-budget scaling experiments justify them.

## 1. Identity and initialization

Artemis is an independently developed foundation model. Official foundation checkpoints begin with random initialization inside this repository. No imported pretrained weights or hidden foreign base model are permitted. See [Core Philosophy](../constitution/core-philosophy.md).

## 2. Current implemented architecture options

The current decoder-only Transformer supports two paths:

- **Legacy-compatible dense path:** full multi-head attention, learned absolute positions, GELU feed-forward block, tied token/output weights. This is preserved for historical checkpoint compatibility.
- **Compact research path:** grouped-query attention (GQA), RoPE, optional per-head QK RMSNorm, optional SwiGLU, and optional factorized token embeddings/output projection.

The compact path passes forward/backward, causal-mask, profile, legacy-checkpoint, and cache-equivalence tests. It has not yet won a matched-quality comparison. The current generator supports incremental KV caching through `forward_with_cache`; Experiment 0008 reports exact cached/uncached output agreement and preliminary 1.27–1.40× speed ratios on a tiny random-initialized CPU smoke model. Those measurements are not representative of trained or larger models. The implementation still expands GQA keys/values temporarily for the current SDPA path, and a preallocated/static cache plus native GQA-kernel benchmarking remain future work.

Research candidates for later:
- Multi-head Latent Attention (MLA) for compact key/value state.
- Sparse Mixture-of-Experts (MoE) only at a scale where routing and communication overhead can be justified.
- Slimmable experts/adaptive compute, recurrent depth, and early exit.
- Memory-efficient attention kernels and sequence packing.

## 3. Tokenizer and language coverage

SentencePiece BPE with byte fallback is the current tokenizer experiment; the byte tokenizer remains available for legacy checkpoints. A production tokenizer is not selected yet.

Evaluate:
- English prose and technical writing.
- Standard Bangla, colloquial Bangla, and Banglish.
- Source code identifiers, punctuation, whitespace, shell commands, paths, and stack traces.
- Mathematics, scientific notation, JSON, and structured tool-call text.

Report fertility, round-trip behaviour, control-token integrity, and per-language sequence cost. Train tokenizer models only on the training split.

## 4. Capability priorities

Software engineering is the first specialist priority: code synthesis, repository comprehension, debugging, test repair, build/CI diagnosis, API integration, code review, and long-horizon changes. General flagship goals remain broad: knowledge, mathematics, reasoning, multilingual dialogue, tool use, factuality, and reliable instruction following.

Do not assume a narrow coding benchmark measures general intelligence. Use the separate capability scorecard in [Frontier Comparison and Evaluation](../evaluation/frontier-targets.md).

## 5. Training curriculum

1. **Pipeline validation:** tokenizer, data, loss mask, checkpoint/resume, and inference tests.
2. **Foundation pretraining:** broad next-token prediction from random initialization using high-quality, licensed, deduplicated general, technical, scientific, code, mathematics, and multilingual data.
3. **Domain-adaptive continued pretraining:** deeper code, maths, technical, and Bangla data while retaining general-domain mixture to reduce forgetting.
4. **Instruction SFT:** reviewed human-authored prompts/responses with response-only loss where appropriate.
5. **Verified capability training:** compiler, unit-test, exact-answer, theorem, and API-schema validated tasks.
6. **Outcome optimisation:** optimise verifiable task outcomes; separately evaluate reward hacking and regressions.
7. **High-quality anneal:** rebalance general language, code, reasoning, and multilingual performance.

The current OASST1 + GSM8K + MBPP run is an early task-SFT experiment from random initialization. It is not broad foundation pretraining.

## 6. Data quality and licence

Every data source needs a versioned manifest with source, revision, licence/permission, attribution, content type, language, hashes, split, preprocessing, deduplication, privacy filtering, and known contamination.

FineWeb is a large general-English candidate, but ODC-By attribution and Common Crawl terms apply, and the source card warns that residual PII and harmful content may remain. It is not a code corpus. For source code, use only a carefully filtered and traceable licence set; maintain original repository/file provenance and follow opt-out removals. For Bangla corpora, review ShareAlike conditions before any weight distribution.

## 7. Scaling methodology

Do not select a headline parameter count first. Compare candidate sizes and architecture configurations at controlled token/FLOP budgets. Fit our own empirical scaling curves for data quality, model size, architecture, and training compute. Use the smallest candidate on the quality/latency/memory Pareto frontier that still meets the full capability requirement.

All reports include total/active parameters, training tokens by source, FLOPs where measurable, hardware, wall time, peak memory, inference latency, tokens/s, per-capability scores, and regressions.

## 8. Checkpoint integrity and release

Every checkpoint records random-init root lineage, seed, source commit, model config, tokenizer hash, data manifest/hash, parent checkpoint hash, optimizer/scheduler state, training phase, compute environment, and evaluation report. Preserve immutable milestone checkpoints and rollback routes.

Frontier capability is an empirical release claim, not a design property. A model cannot be promoted on falling loss or parameter count alone.
