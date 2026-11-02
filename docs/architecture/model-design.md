# Artemis — Model Design and Research Plan

**Status:** Research proposal; no model dimensions are committed.

## 1. Training-from-scratch requirement

Artemis's foundation weights must begin from random initialization. The project may study published architectures, open research, public benchmarks, and external models as references. Those references do not become Artemis's hidden base model.

External models may be used as optional evaluators or comparison points, subject to their terms and the task's data policies. They must not be used to generate unlicensed training corpora or to disguise copied weights as scratch training.

## 2. Initial architecture hypothesis

Start with a **decoder-only Transformer** as the experimental baseline because it is well understood and allows us to validate the complete training pipeline.

Research candidates for later experiments:

- Dense feed-forward blocks as the simplest baseline.
- Sparse Mixture-of-Experts blocks with a learned router.
- Different attention layouts and positional encoding choices.
- Grouped-query or other memory-efficient attention variants.
- KV-cache optimisations for inference.
- Optional multi-token prediction or auxiliary training objectives.

A sparse MoE model may eventually be attractive because it can maintain a large total parameter pool while activating only selected experts for a token. It is not free: routing, load balancing, expert parallelism, communication overhead, and training stability are significant engineering challenges. Shared layers are still used, and a dense model's layer-by-layer computation is not equivalent to fetching a few arbitrary pages from disk.

The decision between dense and MoE must be based on matched-budget experiments: quality, active compute, wall-clock time, memory, communication overhead, and inference throughput.

## 3. Tokenizer and input representation

Evaluate tokenization on representative samples of:
- English prose and technical writing.
- Bengali text and mixed Bengali/English usage.
- Source code, identifiers, punctuation, and whitespace.
- Shell commands, paths, stack traces, and structured data.
- Mathematical notation and common scientific symbols.

Report fertility (tokens per character/word), round-trip behaviour, unknown or pathological cases, code splitting, and language balance. Do not select a tokenizer based only on English benchmark performance.

Tokenizer vocabulary size, algorithm, normalization, and special tokens remain undecided until corpus analysis is complete.

## 4. Capability targets

General-purpose capabilities:
- Reading comprehension and instruction following.
- Knowledge synthesis with calibrated uncertainty.
- Mathematical and scientific problem solving.
- Multilingual communication, including Bengali.
- Planning, abstraction, summarisation, and explanation.
- Structured output and reliable tool-call generation.

Priority capability: software engineering:
- Code comprehension across files and repositories.
- Implementation from specifications.
- Debugging and root-cause analysis.
- Testing, build systems, dependency management, and CI.
- Refactoring and performance reasoning.
- API, database, systems, and infrastructure work.
- Secure code review and vulnerability explanation.
- Long-running work with state tracking and verification.

Capability specialisation should emerge from training and be measured. Do not assume experts will specialise exactly according to human-assigned labels.

## 5. Training stages

1. **Pipeline validation:** tokenizer tests, data loader tests, deterministic small runs, loss monitoring, checkpoint save/resume, and inference.
2. **Small random-init baseline:** train a small dense model to establish the first end-to-end capability baseline.
3. **Foundation pretraining:** next-token prediction over a quality-controlled, licensed, deduplicated corpus.
4. **Capability balancing:** measure and adjust mixture proportions for code, mathematics, technical material, multilingual content, and general language.
5. **Supervised post-training:** instruction following, explanations, structured outputs, and tool-call formats.
6. **Reasoning and task training:** multi-step problem solving, code tasks, debugging traces, and verified solution workflows.
7. **Outcome-based optimisation:** use tests, compiler results, exact answers, and other deterministic outcomes where available. Use human preference evaluation for qualities that are not mechanically verifiable.
8. **Regression and release:** compare candidates with the prior checkpoint on protected evaluations before promotion.

The exact loss functions, optimiser, learning-rate schedule, precision, batch size, context length, token budget, and training mixture must be set by experiments.

## 6. Data quality and provenance requirements

Every dataset component must record:
- Source and acquisition method.
- Licence or documented permission basis.
- Collection date and source revision where available.
- Content type, language, and approximate size.
- Deduplication and filtering procedure.
- Privacy and sensitive-data handling.
- Benchmark-contamination assessment.
- Transformation and preprocessing version.

For source code, track repository origin, licence, language, build metadata when available, test availability, and dependency requirements. A code file is not automatically a high-quality training example.

Do not place secrets, access tokens, personal data without a valid basis, or private repository content into training corpora. Keep data deletion and exclusion mechanisms in the ingestion design.

## 7. Scaling methodology

Use small-to-large scaling experiments rather than choosing a headline parameter count first.

For each experiment, record:
- Model architecture and parameter counts (total and active where relevant).
- Training tokens and data mixture.
- Accelerator type/count and wall-clock time.
- Peak device memory and throughput.
- Optimiser and schedule.
- Training loss and validation loss.
- Evaluation results by capability.
- Inference latency, throughput, and memory.
- Cost and energy estimates where measurable.

A larger model should be approved only when its expected capability gain justifies its training, deployment, and maintenance cost.

## 8. Checkpoints and reproducibility

Each checkpoint must point to:
- Exact source-code revision.
- Training configuration.
- Tokenizer version.
- Dataset manifest and hashes.
- Parent checkpoint, if any.
- Optimiser/scheduler state where needed for resume.
- Random seeds and distributed topology.
- Training logs and evaluation report.

Keep immutable milestone checkpoints and define rollback procedures. Never overwrite the only known-good checkpoint.

## 9. Model release requirements

A candidate must pass:
- Training stability and artifact-integrity checks.
- General and coding evaluation.
- Tool-use and long-horizon task evaluation.
- Factuality and uncertainty calibration tests.
- Regression checks against previous releases.
- Security and policy tests appropriate to the deployment profile.
- Inference resource and latency checks.

Frontier status is an empirical claim, not a design property. Do not claim parity or superiority without comparable evaluation conditions and reproducible results.
