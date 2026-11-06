# Artemus

**Artemus is the complete cognitive-system research project—not a single LLM.** The current `ArtemisLM` implementation is one experimental neural-model component within that larger system. The full Artemus architecture is intended to coordinate perception, world modelling, memory, multiple reasoning methods, planning, action, verification, and controlled learning.

> **Latest documented model-component status (2026-11-06):** A 1,004,928-parameter compact `ArtemisLM` checkpoint from random initialization completed 16,000 task-SFT steps on an OASST1/GSM8K/MBPP mixture. The recovery/evaluation workflow now completes all 16 scripted dialogue turns (16 non-empty, zero context overflows; one old history window trimmed). However, direct math and Python spot checks remain incoherent. This is still an early experimental checkpoint—not a useful general assistant, foundation-pretrained model, or evidence of AGI.

## Mission

Build **Artemus as a general cognitive system** that can perceive inputs, maintain a predictive model of relevant environments, remember experience, select appropriate reasoning methods, form and revise goals, plan, act through controlled interfaces, verify outcomes, and improve through evaluated learning.

A neural language model is one component—not the definition of Artemus. The existing `ArtemisLM` research remains an independently trained model component whose checkpoints begin from random initialization. That provenance rule applies to official ArtemisLM checkpoints; it does not reduce the whole Artemus system to a model.

The first K0 runtime slice now exists in `src/artemus`: a deterministic goal–plan–action–verification loop, mock reasoner, simulated environment, permission gate, and SQLite task-event ledger. This is a small control-system foundation, not a complete cognitive runtime; a predictive world model, richer memory lifecycle, real tool adapters, recovery/replay, multimodal perception, and evaluated learning remain future work. See [Artemus Cognitive System Architecture](docs/architecture/artemus-cognitive-system.md) and the [First Cognitive Kernel milestone](docs/engineering/first-cognitive-kernel.md).

## Core philosophy

Artemus should be candid, intellectually open, direct, and minimally arbitrary in what it discusses. It should avoid viewpoint-based refusals and evasive boilerplate while respecting privacy, consent, access controls, and proportionate safeguards. See the [Artemus Cognitive System Architecture](docs/architecture/artemus-cognitive-system.md) and the [model-component philosophy](docs/constitution/core-philosophy.md).

Scientific papers, public benchmarks, and appropriately licensed data/code may inform our implementation. That is independent research, not copying trained weights. External models may be comparison points or evaluators, not the hidden base model or default training-answer generators.

## System at a glance

- **Artemus:** the complete cognitive system. Its initial K0 kernel slice is implemented; the broader integrated system is not.
- **Cognitive kernel:** a deterministic core for goals, observations, planning proposals, permission checks, actions, independent verification, and SQLite event logging.
- **ArtemisLM:** the existing from-scratch neural language-model experiment; one replaceable component of Artemus.
- **OASIS:** a separately developed execution/runtime capability that may connect through explicit action interfaces.
- **Training and evaluation:** reproducible data preparation, checkpoint lineage, protected holdouts, and regression gates.

Measure ArtemisLM alone separately from the complete Artemus system. Never attribute gains from tools, retrieval, external models, or environment execution to the base-model weights.

## Current experiments

| Experiment | Result | Interpretation |
|---|---|---|
| [0001 — CPU smoke test](docs/experiments/0001-cpu-smoke-test.md) | Tiny random-init training, save/load, and generation passed. | Pipeline validation only. |
| [0002 — OASST1 CPU run](docs/experiments/0002-oasst1-cpu-run.md) | SentencePiece/OASST1 training reduced loss. | Output remained incoherent. |
| [0003 — response-only supervision](docs/experiments/0003-oasst1-response-only.md) | 1.085M parameters, 1,200 CPU steps, response-only target masking. | Output remained incoherent. |
| [0004 — efficient dense smoke](docs/experiments/0004-efficient-dense-smoke.md) | GQA, RoPE, QK norm, and SwiGLU passed architecture/gradient tests. | Candidate mechanics validated, not quality. |
| [0005 — parameter/KV profile](docs/experiments/0005-parameter-profile.md) | 1,250,688 parameters; theoretical FP16 KV state 50% smaller than full MHA at the profiled config. | Parameter/cache accounting only; see Experiment 0008 for the implemented cache. |
| [0006 — factorized embeddings](docs/experiments/0006-factorized-embedding-profile.md) | Reduced the profiled compact candidate to 1,004,928 parameters, about 19.65% fewer than the non-factorized candidate. | No matched-quality result yet. |
| [0007 — multi-task CPU task-SFT](docs/experiments/0007-multitask-cpu-task-sft.md) | 1,347,200-parameter full-MHA model trained 8,000 steps on OASST1/GSM8K/MBPP. | Checkpoint and generation pipeline passed; math/Python outputs remained incoherent. |
| [0008 — incremental KV cache](docs/experiments/0008-incremental-kv-cache.md) | 47 tests passed; cached/uncached outputs matched; three tiny CPU runs measured 1.27–1.40× cached/uncached elapsed-time ratios. | Systems smoke result only; static cache and accelerator benchmarks remain. |
| [0009 — size ladder profile](docs/experiments/0009-size-ladder-profile.md) | Exact profiles for 8.1M, 26.5M, and 88.7M parameter candidates. | Parameter accounting only; no training or capability claim. |

The incremental-cache benchmark is documented in [Experiment 0008](docs/experiments/0008-incremental-kv-cache.md).

The 0007 checkpoint used format v2 and did not embed run metadata. Its sidecar report records the workflow commit, configuration, data manifests, and artifact/checkpoint hashes. New training runs use checkpoint format v3 with provenance metadata.


## General-purpose training programme

The old 16k-step checkpoint is a task-SFT experiment, not the foundation model. The next model-learning phase therefore starts from **fresh random initialization** on broad, pinned pretraining data, then adds reviewed specialist and human-taught data in separate stages.

- [Continuous training programme](docs/training/continuous-training-programme.md): F0 foundation pretraining → F1 domain adaptation → SFT → verified outcomes → human preferences → bounded environment optimisation → multimodal research.
- [Cloud compute options](docs/training/cloud-compute.md): Kaggle/Colab free allocations, institutional compute, and careful paid fallbacks.
- [Kaggle/Colab GPU training notebook](notebooks/kaggle-gpu-training.ipynb): pinned FineWeb-Edu sample, provenance-preserving filtering, tokenizer training, mixed-precision GPU training, evaluation, optional OSCR code adaptation, and export.
- [Machine-readable training curriculum](data/training-curriculum.json): source revisions, stages, licences, and promotion gates.

**Direct human teaching:** install the project and run `artemus-teach --checkpoint <checkpoint.pt>`. You can correct an answer, explicitly approve it, or skip it. Corrections become candidate supervised data with checkpoint/session provenance; they do not silently update the model. Curate them with `artemus-prepare-teaching --input candidates.jsonl --output-dir prepared/`; it keeps whole teaching sessions together across train/validation. Then use the prepared JSONL splits in a separate response-only SFT run after reviewing the samples and manifest.

The new trainer supports independent-document JSONL inputs, one end-of-document token per record, an integrity-checked disk-backed token cache (for larger corpora), gradient accumulation, and CUDA FP16/BF16 mixed precision. GPU training is not run automatically by GitHub Actions: the Kaggle notebook must be opened in an account with an available GPU allocation. No cloud GPU has been provisioned by this repository workflow.

## Neural-model research (ArtemisLM)

The current model-component research direction is documented in:
- [Artemus Cognitive System Architecture](docs/architecture/artemus-cognitive-system.md)
- [First Cognitive Kernel milestone](docs/engineering/first-cognitive-kernel.md)
- [Compact Frontier Flagship Strategy](docs/architecture/compact-frontier-strategy.md)
- [2026 Frontier Research Review](docs/architecture/frontier-research-review.md)
- [Efficient Dense Architecture V2](docs/architecture/efficient-dense-v2.md)
- [Model Design](docs/architecture/model-design.md)
- [Frontier Comparison and Evaluation Plan](docs/evaluation/frontier-targets.md)
- [Flagship Capability Roadmap](docs/evaluation/capability-roadmap.md)
- [Training Strategy](docs/training/training-strategy.md)
- [Data Source Registry](docs/data/source-registry.md)
- [Additional Data Sources and Ingestion Priorities](docs/data/next-sources.md)

The strategy is informed by architecture-aware scaling laws, data-quality scaling, compute/data trade-offs, multilingual ATLAS results, small-model reasoning studies, and modern sparse-expert research. Those papers guide experiments; they do not guarantee that a small model can match closed frontier systems.

## Data and training policy

Use version-pinned, licence-reviewed, deduplicated sources with source-level provenance. Keep protected benchmark splits out of training. FineWeb is a potential broad English pretraining source, not a code corpus; its ODC-By attribution requirements, Common Crawl terms, and residual PII/content risks require additional processing. Code datasets retain original repository/file licences and removal obligations. Bangla datasets with ShareAlike conditions require review before weight distribution.

The current OASST1/GSM8K/MBPP mixture is an early **task-SFT** experiment. It is not the broad next-token foundation-pretraining corpus required for a frontier model.

## GitHub Actions

- `validation.yml`: unit/data-integrity tests and Python syntax checks.
- `efficient-dense-ci.yml`: compact architecture smoke training, parameter profiling, cache correctness, and cached/uncached CPU benchmarking.
- `capability-training.yml`: manual CPU continuation plus a scheduled dialogue-health check for the historical OASST1/GSM8K/MBPP task-SFT checkpoint. It no longer starts a full CPU training run on every model-code push.

The 16,000-step checkpoint and successful context-trimmed dialogue evaluation are archived in [GitHub Actions run 42233796727](https://github.com/FAHTHIZ/Artemus/actions/runs/42233796727). The checkpoint can be downloaded from that run's artifact. It remains incoherent on the recorded mathematics and Python spot checks; use the new F0 notebook for broad pretraining rather than repeating this same task mixture.

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

1. Extend K0 into explicit task-state transitions, cancellation, recovery, and deterministic replay.
2. Extend episodic memory with provenance, scope, retention, query, and deletion controls.
3. Build a versioned world-state model that distinguishes observations, inferences, predictions, and contradictions.
4. Add explicit goal graphs, bounded planning, permission-checked actions, and independent verification.
5. Connect ArtemisLM behind a replaceable reasoning adapter; benchmark base-model and full-system results separately.
6. Build protected evaluations for generalisation, reasoning, mathematics, coding, multilingual use, memory, long-horizon work, and reliability.
7. Run the pinned FineWeb-Edu GPU foundation-pretraining pilot, then add OSCR/Bangla/math/code sources under their own provenance and licensing reviews.
8. Make any frontier or AGI capability claim only after broad, independent, reproducible evidence.

## Repository hygiene

Datasets, secrets, local caches, experiment outputs, and model checkpoints must not be committed. Keep large artifacts in an explicitly configured storage location with versioning, access controls, provenance manifests, and integrity hashes.
