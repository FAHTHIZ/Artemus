# Artemus

**Artemus is a from-scratch foundation-model and agent-systems project.** Its intended model, Artemis, is a general-purpose assistant with exceptional software-engineering capability.

> **Current status:** Architecture documentation and a tiny CPU training baseline are committed. Experiment 0001 completed 250 training steps and verified checkpoint reload and generation. This remains a pipeline smoke test—not a capable assistant or frontier-model training run.

## Mission

Build an independently trained model whose weights begin from random initialization—not a renamed or fine-tuned copy of an existing frontier model—and develop it through reproducible training, evaluation, and controlled iteration.

The long-term ambition is a compact frontier flagship with strong coding, debugging, repository understanding, mathematics, general reasoning, multilingual capability, tool use, and long-horizon task execution. This is an objective to measure, not a capability claim or guarantee. The final parameter budget will be chosen by measured quality/compute/memory scaling curves rather than a fixed headline number.

## System at a glance

- **Artemis:** The foundation model and its inference interface.
- **OASIS:** The agent runtime responsible for planning, tools, memory, execution, and verification.
- **Training platform:** Data preparation, tokenizer experiments, pretraining, post-training, checkpoints, and experiment tracking.
- **Evaluation platform:** Independent tests, regression gates, coding benchmarks, and real-project task evaluations.
- **Knowledge and memory services:** Retrieval over approved external information and durable project/session state.

The model, agent runtime, training platform, and external memory are separate components with explicit interfaces. See [the system architecture](docs/architecture/system-overview.md).

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

## Run locally

```bash
python -m pip install -e ".[dev]"
pytest
artemus-train --data examples/demo_corpus.txt --steps 20 --batch-size 4 --block-size 32 --layers 2 --heads 2 --embedding-size 32 --eval-interval 10 --output artifacts/artemis-tiny.pt
artemus-generate --checkpoint artifacts/artemis-tiny.pt --prompt "Artemis is" --tokens 80
```

The byte tokenizer and tiny model are temporary research baselines, not the final architecture or a useful assistant.

## Next milestones

1. Complete and evaluate the current small multi-task CPU experiment without claiming frontier capability.
2. Run matched baseline versus factorized-embedding/GQA architecture comparisons on identical data and token budgets.
3. Implement and benchmark incremental KV caching; measure prefill/decode latency and peak memory.
4. Build the licensed broad pretraining corpus and a protected evaluation suite, including Bangla/Banglish and code.
5. Obtain appropriately sized accelerator compute for genuine foundation pretraining and controlled size/data scaling sweeps.
6. Develop OASIS separately and measure base-model versus agent-system performance.

## Repository hygiene

Datasets, credentials, local caches, experiment outputs, and model checkpoints must not be committed. Large assets belong in an explicitly configured data or artifact store with access controls and provenance records.

## Naming

The GitHub repository is named **Artemus**. **Artemis** is the working name used in the design documents for the model/system. The repository name is retained as-is for now.
