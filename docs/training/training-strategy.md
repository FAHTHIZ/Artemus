# Artemis — Training Strategy

**Status:** Active staged plan. Compute-dependent choices remain open until measured.

## 1. Non-negotiable training lineage

The foundation model starts from random weights in this project. Training checkpoints preserve initialization seed, code commit, data/tokenizer hashes, parent checkpoint, full run configuration, and evaluation evidence. External models may be reference evaluators, never hidden foundation weights or default generators of training answers.

## 2. Data pipeline

```text
SOURCE + IMMUTABLE REVISION
  -> LICENCE / PERMISSION REVIEW
  -> ACQUISITION + HASH
  -> FORMAT NORMALISATION
  -> LANGUAGE / DOMAIN / QUALITY LABELS
  -> PRIVACY / PII FILTERING
  -> EXACT + NEAR-DUPLICATE DETECTION
  -> BENCHMARK-CONTAMINATION REVIEW
  -> SOURCE-AWARE SPLITS
  -> MIXTURE / SAMPLING POLICY
  -> VERSIONED MANIFEST
  -> TOKENIZER / TRAINING
```

Every source and transformation must be reproducible. Rejected and quarantined records keep exclusion reasons.

## 3. Data families

The future foundation mix should include:
- High-quality general prose and reference material.
- Educational, scientific, and mathematical text.
- Technical documentation and tutorials.
- Source code from repositories with compatible licences and file-level provenance.
- Bangla, colloquial Bangla, Banglish, and English.
- Human-authored conversational and instruction data.
- Programmatically generated tasks with deterministic, independent validators.

FineWeb is a large general-English pretraining candidate. It is not enough by itself: its ODC-By 1.0 attribution requirements and Common Crawl terms must be respected, residual PII/harmful-content risks need extra filtering, and it contains relatively little code. The Stack and other code collections require original-file licence/provenance handling and opt-out updates. Bangla corpora with CC-BY-SA terms need a distribution/legal review before their use in a released model.

See [Additional Data Sources](../data/next-sources.md).

## 4. Training stages

### Stage A — pipeline correctness
Tokenizer behaviour, data loaders, split integrity, response masks, deterministic tiny training, save/load, resume, and generation. These are tests, not capability milestones.

### Stage B — broad foundation pretraining
Train next-token prediction from random initialization on a large, quality-controlled corpus. Keep held-out language modelling, document, code, maths, and multilingual evaluations separate. Record unique tokens separately from repeated tokens.

### Stage C — domain-adaptive continued pretraining
Increase exposure to high-value technical, coding, mathematical, scientific, and target-language data. Preserve general-domain mixture to reduce catastrophic forgetting. Measure per-domain validation loss and task outcomes.

### Stage D — supervised instruction tuning
Use reviewed human-authored and appropriately licensed instructions, high-quality answers, and multi-turn conversations. Use response-only masks where appropriate; keep complete conversation trees in one split.

### Stage E — verified capability training
Train on code, math, and tool tasks with exact checks: compilers, unit tests, theorem/exact solvers, numerical validators, and API schemas. Execute model code in isolated, network-disabled sandboxes with strict limits.

### Stage F — outcome optimisation
Use verifiable success signals, preference data with documented origin, and explicit reward-hacking tests. Do not reward superficial answer length or benchmark leakage.

### Stage G — high-quality anneal and release
Balance general language, code, reasoning, mathematics, and multilingual quality. Promote only after full regression, factuality, privacy, and capability evaluation.

## 5. Current experimental lane

The OASST1 + GSM8K + MBPP run is a small multi-task task-SFT experiment from random initialization. GSM8K official test data and MBPP protected test/prompt task IDs remain excluded. It validates data plumbing and training mechanics. It is not the foundation pretraining stage and cannot produce frontier-level capability at the present CPU/parameter scale.

The manual workflow is `.github/workflows/capability-training.yml`; quick unit tests are separate from expensive training.

## 6. Evaluation and data integrity

Maintain train, validation, and protected test splits. Use source-level or repository-level separation where possible. Keep benchmark tests out of tokenizer fitting, prompt examples, synthetic task generation, and all training phases. Track likely benchmark contamination.

A benchmark win on public data is not enough. Keep private newly authored holdout tasks and use multiple trials where generation is stochastic.

## 7. Compute and reproducibility

GitHub-hosted CPU runners are for tests, tokenizer/data preparation, profile calculations, and tiny smoke runs. They are not adequate for frontier-scale foundation pretraining. A serious training phase needs accelerator infrastructure, efficient packed batches, mixed precision, checkpoint/resume for model+optimizer+scheduler+RNG+data position, monitoring, and explicit compute/cost budgets.

Do not assume a free GPU quota. The phone can control repository work and review; the hosted runner is the compute currently used by these experiments.

## 8. Interaction feedback loop

```text
INTERACTION
 -> CONSENTED / POLICY-COMPLIANT CAPTURE
 -> FAILURE OR SUCCESS DIAGNOSIS
 -> VERIFIED CORRECTION
 -> CURATED TRAINING EXAMPLE
 -> CANDIDATE RUN
 -> HELD-OUT + REGRESSION EVALUATION
 -> APPROVAL / REJECTION
 -> VERSIONED CHECKPOINT
```

Memory updates and weight updates are different. Never rewrite production weights directly from live conversation feedback.

## 9. Tokenizer and Bangla efficiency experiments

A compact flagship cannot waste context on avoidable token fragmentation. Evaluate tokenizer candidates on held-out, source-balanced samples of:
- English prose and technical documentation.
- Standard Bangla, colloquial Bangla, and Banglish/code-switching.
- Python and other target programming languages.
- Mathematics, formulas, shell commands, and structured tool payloads.

Report tokens per whitespace word, tokens per Unicode character, byte fallback rate, unknown/rare-fragment rate, and sequence-length distributions. Also measure round-trip correctness and downstream task quality; lower token count alone is not a quality win.

Compare SentencePiece BPE and Unigram baselines before considering morphology-aware segmentation. The B-CORE research release describes morphology-aware tokenization and principled denoising for Bangla; treat this as a reproducible research hypothesis, not a requirement to import its model weights. Keep tokenizer training restricted to training data and version every tokenizer with its corpus manifest.

B-CORE is a high-value Bangla pretraining candidate, but its CC-BY-4.0 dataset card notes that individual source components may have their own attribution obligations. Audit the component manifest before use or redistribution. See [Additional Data Sources](../data/next-sources.md).


## 10. Foundation-scale trainer requirements

The current trainer is deliberately small and simple. Before large-corpus pretraining, replace its whole-text/in-memory token loading path with a scalable input system:

1. **Versioned token shards:** stream documents through privacy/quality/dedup filters and a pinned tokenizer; write packed, memory-mapped token shards with document-boundary markers and a source-to-shard manifest.
2. **Deterministic sampling:** use shard-aware shuffling and a seeded buffer; record shard order, offsets, sample weights, and data position so a resumed run does not silently repeat or skip a different data distribution.
3. **Efficient packing:** pack sequences to the model context window while preserving document boundaries and preventing cross-document target leakage.
4. **Training schedule:** add warmup plus cosine decay, configurable weight decay/betas, gradient accumulation, gradient clipping, and explicit token-based schedules. Compare optimizers under matched FLOPs rather than switching based on one loss curve.
5. **Hardware-aware execution:** add BF16/FP16 where supported, fused optimizers/kernels, activation checkpointing, and benchmarked batch/sequence shapes. Keep a numerically stable FP32 reference.
6. **Exact resume:** checkpoint model, optimizer, scheduler, scaler, RNG states, data/shard cursor, tokenizer hash, and full config. Validate resume equivalence against an uninterrupted short run.
7. **Measurement:** log unique tokens, repeated token presentations, tokens/sec, FLOPs estimate, peak memory, wall-clock time, energy/cost where available, and held-out capability outcomes.

Do not scale the current 8K-step CPU task-SFT script by simply increasing the step count. Build the data and systems path first, then run controlled size/data/compute experiments on suitable accelerators.

