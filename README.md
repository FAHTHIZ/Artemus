# Artemus

**Artemus is a from-scratch foundation-model and agent-systems project.** Its intended model, Artemis, is a general-purpose assistant with exceptional software-engineering capability.

> **Current status:** Experiments 0001–0003 validated the initial pipeline, SentencePiece tokenization, and response-only loss masking. Experiment 0003 completed 1,200 CPU steps on a 1,085,056-parameter random-init model; its sample output remains incoherent. The next CPU experiment is currently training from random initialization on a pinned mixture of up to 30,000 OASST1 conversations plus GSM8K train-split math and MBPP official training-ID code tasks. All unit tests, source checksums, preprocessing, and mixture construction have passed; training and generation evaluation are still in progress. Artemis is not yet a usable assistant. See the [Experiment 0003 report](docs/experiments/0003-oasst1-response-only.md).

## Mission

Build an independently trained model whose weights begin from random initialization—not a renamed or fine-tuned copy of an existing frontier model—and develop it through reproducible training, evaluation, and controlled iteration.

The long-term ambition is frontier-class general intelligence with especially strong coding, debugging, repository understanding, mathematics, tool use, and long-horizon task execution. This is an objective to measure, not a capability claim or guarantee.

## System at a glance

- **Artemis:** The foundation model and its inference interface.
- **OASIS:** The agent runtime responsible for planning, tools, memory, execution, and verification.
- **Training platform:** Data preparation, tokenizer experiments, pretraining, post-training, checkpoints, and experiment tracking.
- **Evaluation platform:** Independent tests, regression gates, coding benchmarks, and real-project task evaluations.
- **Knowledge and memory services:** Retrieval over approved external information and durable project/session state.

The model, agent runtime, training platform, and external memory are separate components with explicit interfaces. See [the system architecture](docs/architecture/system-overview.md).

## Core philosophy

Artemis is an independently trained model whose foundation weights begin from random initialization. No renamed or hidden pretrained checkpoint is allowed. The project optimises capability per parameter and unit of compute, and treats frontier-level capability as a measurable ambition rather than a claim. It aims for candid, intellectually open responses without arbitrary viewpoint-based refusals, while preserving explicit privacy, consent, legal, and serious-harm boundaries.

See [Core Philosophy](docs/constitution/core-philosophy.md), [Compact Frontier-Capability Strategy](docs/architecture/compact-frontier-strategy.md), and [Frontier Comparison Plan](docs/evaluation/frontier-targets.md).

## Design principles

1. **Train from scratch.** Foundation weights start from random initialization. Existing models may be used as external evaluators or research references, never silently substituted as Artemis.
2. **Measure capability, not parameter count.** Scale only when experiments show that additional compute produces meaningful gains.
3. **Retrieve selectively.** Use context construction, search, structured indexes, and memory instead of trying to load an entire knowledge store into every request.
4. **Verify work.** Tool execution, code changes, and factual claims should be checked wherever practical.
5. **Improve deliberately.** Conversations can generate candidate lessons; model-weight updates require curated data, evaluation, regression checks, and rollback.
6. **Be candid and intellectually open.** Avoid arbitrary corporate or viewpoint-driven evasiveness. State uncertainty honestly. Tool permissions and safeguards remain in force.
7. **Keep the system reproducible.** Version datasets, configurations, code, checkpoints, and evaluation results.

## Documentation

- [System architecture](docs/architecture/system-overview.md)
- [Model design and research plan](docs/architecture/model-design.md)
- [OASIS agent runtime](docs/architecture/oasis-runtime.md)
- [Training strategy](docs/training/training-strategy.md)
- [Evaluation and release gates](docs/evaluation/quality-gates.md)
- [Engineering roadmap](docs/engineering/roadmap.md)
- [First training run](docs/engineering/first-training-run.md)
- [Experiment 0001 report](docs/experiments/0001-cpu-smoke-test.md)
- [Experiment 0002 — OASST1 CPU run](docs/experiments/0002-oasst1-cpu-run.md)
- [Experiment 0003 — response-only supervision](docs/experiments/0003-oasst1-response-only.md)
- [Current capability-mixture training run](https://github.com/FAHTHIZ/Artemus/actions/runs/41372200736)
- [Training data source registry](docs/data/source-registry.md)
- [Additional datasets and ingestion priorities](docs/data/next-sources.md)
- [Core philosophy](docs/constitution/core-philosophy.md)
- [Compact frontier strategy](docs/architecture/compact-frontier-strategy.md)
- [Frontier comparison plan](docs/evaluation/frontier-targets.md)

## Run locally

```bash
python -m pip install -e ".[dev]"
pytest
artemus-train --data examples/demo_corpus.txt --steps 20 --batch-size 4 --block-size 32 --layers 2 --heads 2 --embedding-size 32 --eval-interval 10 --output artifacts/artemis-tiny.pt
artemus-generate --checkpoint artifacts/artemis-tiny.pt --prompt "Artemis is" --tokens 80
```

The byte tokenizer and tiny model are temporary research baselines, not the final architecture or a useful assistant.

## Initial next steps

1. Record the available workstation hardware and storage.
2. Select the development environment and accelerator backend.
3. Build a properly sourced and licensed corpus and a held-out evaluation suite.
4. Build OASIS as a separately testable runtime.
5. Run controlled architecture and scaling experiments when sufficient compute is available.

## Repository hygiene

Datasets, credentials, local caches, experiment outputs, and model checkpoints must not be committed. Large assets belong in an explicitly configured data or artifact store with access controls and provenance records.

## Naming

The GitHub repository is named **Artemus**. **Artemis** is the working name used in the design documents for the model/system. The repository name is retained as-is for now.
