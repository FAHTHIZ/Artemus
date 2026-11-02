# Artemus

**Artemus** is the repository name. **Artemis** is the independently developed foundation model and flagship assistant being built here.

> **Current status (2026-11-02):** The training pipeline, data-ingestion checks, checkpoint loading, and compact-architecture smoke tests work. The 1.085M-parameter response-only model from Experiment 0003 was incoherent. Experiment 0007 completed 8,000 CPU steps on an OASST1/GSM8K/MBPP multi-task mixture, but its math and Python generations were still incoherent. The newer 1.25M compact GQA/RoPE/QKNorm/SwiGLU candidate and 1.005M factorized-embedding candidate pass architecture tests and reduce parameter counts, but have not yet passed matched-quality capability comparisons. **Artemis is not yet a usable assistant or a frontier-capable model.**

## Mission

Build a compact, independently trained foundation model with frontier-level ambition across software engineering, reasoning, mathematics, general knowledge, multilingual communication, and long-horizon tool use.

Every official foundation checkpoint begins from random initialization in this project. Artemis is not a renamed model, a fine-tune of another provider's checkpoint, or a hidden import of someone else's learned weights.

The final parameter budget is intentionally undecided. We will choose it from measured scaling curves and a capability/quality/latency/memory/cost Pareto frontier—not from a headline parameter target.

## Core philosophy

Artemis should be candid, intellectually open, direct, and minimally arbitrary in what it discusses. It should avoid viewpoint-based refusals and evasive boilerplate. This does not remove privacy, consent, legal-access, or proportionate serious-harm boundaries. See the [Core Philosophy](docs/constitution/core-philosophy.md).

Scientific papers, public benchmarks, and appropriately licensed data/code may inform our implementation. That is independent research, not copying trained weights. External models may be comparison points or evaluators, not the hidden base model or default training-answer generators.

## System at a glance

- **Artemis:** independently trained foundation model and inference interface.
- **OASIS:** separate agent runtime for planning, tools, memory, execution, and verification.
- **Training platform:** data preparation, tokenizers, pretraining, post-training, checkpoint lineage, and experiment tracking.
- **Evaluation platform:** private holdouts, capability-specific benchmarks, regression gates, and coding task execution.
- **Knowledge and memory services:** approved-source retrieval and durable task/session state.

Base-model capability and Artemis + OASIS system performance are measured separately.

## Current experiments

| Experiment | Result | Interpretation |
|---|---|---|
| [0001 — CPU smoke test](docs/experiments/0001-cpu-smoke-test.md) | Tiny random-init training, save/load, and generation passed. | Pipeline validation only. |
| [0002 — OASST1 CPU run](docs/experiments/0002-oasst1-cpu-run.md) | SentencePiece/OASST1 training reduced loss. | Output remained incoherent. |
| [0003 — response-only supervision](docs/experiments/0003-oasst1-response-only.md) | 1.085M parameters, 1,200 CPU steps, response-only target masking. | Output remained incoherent. |
| [0004 — efficient dense smoke](docs/experiments/0004-efficient-dense-smoke.md) | GQA, RoPE, QK norm, and SwiGLU passed architecture/gradient tests. | Candidate mechanics validated, not quality. |
| [0005 — parameter/KV profile](docs/experiments/0005-parameter-profile.md) | 1,250,688 parameters; theoretical FP16 KV state 50% smaller than full MHA at the profiled config. | No incremental KV cache or measured decode-speed benefit yet. |
| [0006 — factorized embeddings](docs/experiments/0006-factorized-embedding-profile.md) | Reduced the profiled compact candidate to 1,004,928 parameters, about 19.65% fewer than the non-factorized candidate. | No matched-quality result yet. |
| [0008 — incremental KV cache](docs/experiments/0008-incremental-kv-cache.md) | 47 tests passed; cached and uncached logits/outputs match. Tiny CPU run was 1.33× faster with cache. | Systems smoke result only; static cache and accelerator benchmarks remain. |
| [0007 — multi-task CPU task-SFT](docs/experiments/0007-multitask-cpu-task-sft.md) | 1,347,200-parameter full-MHA model trained 8,000 steps on OASST1/GSM8K/MBPP. | Checkpoint and generation pipeline passed; math/Python outputs remained incoherent. |

The 0007 checkpoint used format v2 and did not embed run metadata. Its sidecar report records the workflow commit, configuration, data manifests, and artifact/checkpoint hashes. New training runs use checkpoint format v3 with provenance metadata.

## Frontier flagship research

Our current research direction is documented in:
- [Compact Frontier Flagship Strategy](docs/architecture/compact-frontier-strategy.md)
- [2026 Frontier Research Review](docs/architecture/frontier-research-review.md)
- [Efficient Dense Architecture V2](docs/architecture/efficient-dense-v2.md)
- [Model Design](docs/architecture/model-design.md)
- [Frontier Comparison and Evaluation Plan](docs/evaluation/frontier-targets.md)
- [Training Strategy](docs/training/training-strategy.md)
- [Data Source Registry](docs/data/source-registry.md)
- [Additional Data Sources and Ingestion Priorities](docs/data/next-sources.md)

The strategy is informed by architecture-aware scaling laws, data-quality scaling, compute/data trade-offs, multilingual ATLAS results, small-model reasoning studies, and modern sparse-expert research. Those papers guide experiments; they do not guarantee that a small model can match closed frontier systems.

## Data and training policy

Use version-pinned, licence-reviewed, deduplicated sources with source-level provenance. Keep protected benchmark splits out of training. FineWeb is a potential broad English pretraining source, not a code corpus; its ODC-By attribution requirements, Common Crawl terms, and residual PII/content risks require additional processing. Code datasets retain original repository/file licences and removal obligations. Bangla datasets with ShareAlike conditions require review before weight distribution.

The current OASST1/GSM8K/MBPP mixture is an early **task-SFT** experiment. It is not the broad next-token foundation-pretraining corpus required for a frontier model.

## GitHub Actions

- `validation.yml`: quick tests and syntax checks on changes.
- `efficient-dense-ci.yml`: compact architecture smoke training and parameter/KV profiling.
- `capability-training.yml`: manual `workflow_dispatch` for the longer multi-task CPU experiment.

The 8,000-step completed run is recorded at [GitHub Actions run 41372200736](https://github.com/FAHTHIZ/Artemus/actions/runs/41372200736). The artifact is retained until 2026-11-09.

## Run locally

```bash
python -m pip install -e ".[dev]"
pytest -q
artemus-train \
  --data examples/demo_corpus.txt \
  --steps 20 \
  --batch-size 4 \
  --block-size 32 \
  --layers 2 \
  --heads 2 \
  --embedding-size 32 \
  --eval-interval 10 \
  --output artifacts/artemis-tiny.pt
```

The byte tokenizer, tiny demo corpus, and smoke model are engineering baselines—not the final training recipe or a useful assistant.

## Next milestones

1. Add protected held-out evaluation for math, code, general language, and Bangla/Banglish.
2. Compare dense and factorized candidates under identical tokenizer, data, seed, and training-token budgets.
3. Implement and benchmark an actual incremental KV cache.
4. Build a large, licensed, deduplicated foundation-pretraining mixture and audit its quality.
5. Secure suitable accelerator compute for true foundation pretraining and size/data/architecture scaling experiments.
6. Train and evaluate the base model through pretraining, SFT, verified task training, and outcome optimisation.
7. Develop OASIS separately and report agent-system performance independently.
8. Claim frontier parity only after comparable, independent results across the capability scorecard.

## Repository hygiene

Datasets, secrets, local caches, experiment outputs, and model checkpoints must not be committed. Keep large artifacts in an explicitly configured storage location with versioning, access controls, provenance manifests, and integrity hashes.
