# Compact Frontier-Capability Strategy

**Status:** Research direction; all architecture decisions require controlled ablations.  
**Invariant:** Foundation weights start from random initialization inside Artemis.

## 1. Objective

Maximise capability per stored parameter, active parameter, FLOP, byte of memory, watt, and unit of training compute.

The long-term ambition is frontier-class performance across coding, reasoning, mathematics, languages, and knowledge work with a much smaller deployed model. This depends on data, compute, optimisation, post-training, inference-time compute, tools, and rigorous evaluation—not a clever block alone.

## 2. Current evidence

- Experiment 0001 proved the initial random-init training loop can run, save, reload, and generate.
- Experiment 0002 used OASST1, SentencePiece, 1,085,056 parameters, and 1,200 CPU steps. Loss fell, but generated text remained incoherent.
- Experiment 0003 is testing assistant-response-only supervision.

These are pipeline experiments, not meaningful foundation pretraining. More steps on the same tiny setup will not create a frontier assistant.

## 3. Research evidence and limits

### ThinkSLM (EMNLP 2025)
The study evaluated 72 small language models over 17 reasoning benchmarks and reports that training method and data quality strongly influence small-model reasoning. It also identifies robustness gaps that remain more common in larger models.
Paper: https://aclanthology.org/2025.emnlp-main.1659/

### HobbyLM (October 2026)
The authors report a from-scratch sparse model with 500M total parameters and 169M active parameters, trained on 40B tokens, and a separate 1.04B-total / 305M-active model trained on a curated 100B-token mixture. This is a useful engineering reference, not evidence that a sub-billion model matches current frontier systems. Its token and compute scale is far beyond our current free CPU runs.
Paper: https://arxiv.org/abs/2610.34582

## 4. Architecture hypotheses

Change one variable per experiment. Establish a reproducible dense baseline before adding sparse routing.

### A. Efficient dense baseline
Ablate:
- RMSNorm and stable initialisation.
- RoPE or another position scheme with measured long-context behaviour.
- Grouped-query attention (GQA) to reduce KV-cache memory.
- Fused/Flash attention where supported.
- SwiGLU feed-forward blocks.
- Tied token embedding/output weights where evaluation supports it.
- Context packing with explicit document/conversation boundaries.
- Schedules, warmup, weight decay, and gradient clipping.

### B. Sparse expert model
After a trained dense baseline exists, test fine-grained Mixture-of-Experts:
- Shared attention and routing layers.
- Several feed-forward experts.
- Top-k or sigmoid routing with measured load balance.
- Router-loss and expert-capacity ablations.
- Total versus active parameters, memory, FLOPs, tokens/s, and wall-clock.

MoE can reduce compute per token for a given total capacity, but stored weights and optimizer state do not disappear. Routing, communication, and memory overhead can make it slower or more expensive on a single CPU.

### C. Recurrent depth and adaptive compute
Investigate weight-shared blocks or additional reasoning passes only after the baseline is stable. Measure whether extra compute improves hard-task accuracy enough to justify latency. Repeated layers are not a free parameter multiplier.

### D. Inference-time system
Keep raw model capability separate from OASIS-assisted capability. OASIS may add repository indexing, retrieval, test execution, compiler feedback, search over approved sources, task planning, and verification. Report base-model and model-plus-tools results separately.

## 5. Training lifecycle

1. **Data audit and tokenizer:** evaluate English, Bengali, Banglish, source code, shell, maths, and structured text; measure fertility, round-trip behaviour, and control-token integrity.
2. **Foundation pretraining:** next-token prediction from random weights over a balanced, deduplicated, licensed mixture of general, technical, code, mathematics, scientific, and multilingual data. OASST1 alone is not a foundation corpus.
3. **Supervised instruction tuning:** reviewed prompt/response examples, explicit role markers, response-only loss, and group-separated validation.
4. **Verified capability training:** deterministic or human-authored tasks validated by compilers, unit tests, exact solvers, theorem checkers, and sandboxed execution.
5. **Outcome optimisation:** use verifiable task success and monitor reward hacking.
6. **High-quality data anneal:** final phase emphasising high-quality diverse material; record the mixture and token count.

External frontier-model outputs are not default training labels. Human-created data and deterministic, independently verified synthetic tasks preserve the project's independent lineage.

## 6. Matched-budget parameter experiments

Do not lock a headline parameter count. Compare small and medium dense candidates with a sparse candidate at similar active compute.

Each experiment records total and active parameters; tokens by source and mixture; accelerator and wall time; estimated FLOPs; peak training and inference memory; tokens/s; held-out scores by capability; quality per active parameter, FLOP, byte, and cost; and multiple seeds/uncertainty where practical.

A smaller model wins only if it retains the quality that matters under comparable evaluations and resource constraints.

## 7. Compute reality

GitHub public CPU runners are useful for tests, data processing, tokenizer experiments, and very small runs. They are not a practical path to frontier-scale foundation pretraining.

Free GPU notebook quotas may help for occasional experiments, but they are variable and not guaranteed continuous infrastructure. Training must checkpoint, resume, and tolerate interruptions.

No result is a frontier result until data scale, compute, architecture, and evaluation justify that statement.

## 8. Milestones

1. Correct reproducible training pipeline.
2. Coherent small model on held-out prompts.
3. Useful narrow capability (initially coding) with verifiable tests.
4. Balanced general, coding, mathematics, and multilingual ability.
5. Efficient dense/sparse architecture comparison.
6. Model-plus-OASIS long-horizon repository evaluation.
7. Frontier-comparison report under comparable benchmarks and conditions.

The first useful milestone is a small independently trained model that answers held-out questions coherently and solves unseen coding tasks at a measured success rate—not a headline parameter count.
