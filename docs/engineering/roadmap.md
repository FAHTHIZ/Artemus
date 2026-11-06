# Artemus — Engineering Roadmap

**Status:** Initial model-and-runtime roadmap; dates are not estimated until hardware and available compute are known.

> **System-level priority:** This roadmap's original model/runtime stages do not implement the full Artemus cognitive system. Use [Artemus Cognitive System Architecture](../architecture/artemus-cognitive-system.md) and [First Cognitive Kernel milestone](first-cognitive-kernel.md) for the new system-first implementation sequence.

## Guiding rule

Prove each layer at small scale before expanding it. The first model is a pipeline validation milestone, not the final capability target.

## Stage 0 — Project specification (current)

Deliverables:
- Architecture boundaries and invariants.
- Model research plan.
- OASIS runtime contract.
- Data/training lifecycle.
- Evaluation and release gates.

Exit criteria:
- Hardware inventory collected.
- Naming and initial development environment confirmed.
- Initial dataset-source and licensing policy agreed.

## Stage 1 — Reproducible development environment

Deliverables:
- Python/ML environment selected based on the machine and supported accelerator.
- Repository package layout and configuration conventions.
- Unit-test and formatting/linting setup.
- Structured logging and experiment tracking.
- Artifact and dataset storage plan.

Exit criteria:
- A clean setup can reproduce the same smoke test.
- No secrets or large artifacts are committed to Git.

## Stage 2 — Minimal random-init training baseline

Deliverables:
- Tokenizer experiment and tests.
- Minimal training data loader.
- Small decoder-only Transformer.
- Training loop, validation loss, checkpoint save/resume.
- Basic inference script and smoke tests.

Exit criteria:
- Model weights are demonstrably random-initialized.
- Training loss behaves as expected on a small controlled corpus.
- Checkpoint can be resumed and inference works.
- Run metadata is captured.

## Stage 3 — Data and evaluation foundations

Deliverables:
- Dataset manifests and source provenance.
- Deduplication and filtering pipeline.
- Initial general, code, math, and multilingual evaluation sets.
- Contamination and regression management.

Exit criteria:
- A training run can be reproduced from its recorded manifest.
- Evaluation is separated from training data.

## Stage 4 — First useful Artemis checkpoint

Deliverables:
- Larger baseline selected based on hardware.
- Curated pretraining mixture.
- Instruction/post-training data format.
- Coding and general capability evaluation.

Exit criteria:
- Measurable improvement over the tiny baseline.
- Known failure modes are documented.

## Stage 5 — OASIS minimum viable runtime

Deliverables:
- Model adapter (mocked until inference API is ready).
- Planner and task-state machine.
- Typed tool registry.
- Sandboxed workspace execution.
- Verification hooks and audit records.

Exit criteria:
- One end-to-end repository task can be planned, executed, tested, and reported with evidence.

## Stage 6 — Memory and retrieval

Deliverables:
- Document and code indexing.
- Lexical/vector/structural search.
- Scoped session and project memory.
- Provenance, retention, and deletion controls.

Exit criteria:
- Retrieval improves task success or context efficiency in controlled evaluation.

## Stage 7 — Post-training and agent improvement

Deliverables:
- Supervised instruction training.
- Tool-use and verified coding trajectories.
- Outcome-based optimisation experiments.
- Candidate promotion and rollback.

Exit criteria:
- Improvements are measurable and regressions are controlled.

## Stage 8 — Scaling research

Deliverables:
- Dense scaling baselines.
- MoE experiments only if justified.
- Distributed training and sharded checkpoints.
- Inference throughput and memory optimisation.

Exit criteria:
- A larger architecture earns adoption through measured quality/compute efficiency.

## Stage 9 — Continuous release cycle

Every release follows:
1. Freeze candidate checkpoint and configuration.
2. Run the protected evaluation suite.
3. Review regressions and known limitations.
4. Promote or reject.
5. Publish a reproducible model card and release report.
6. Preserve the prior stable checkpoint.

## Immediate blocker list

- Workstation specifications are not yet known.
- Available training compute and budget are not yet known.
- Data-source/licensing plan needs confirmation.
- Model dimensions, tokenizer, and context length need baseline experiments.
