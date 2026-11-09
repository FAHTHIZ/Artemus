# Artemis — Evaluation and Quality Gates

## 1. Purpose

Evaluation is a release gate, not a marketing layer. A checkpoint is not an improvement until it demonstrates gains under repeatable conditions without unacceptable regressions.

## 2. Evaluation dimensions

| Dimension | What to test |
|---|---|
| English language | Comprehension, instruction following, summarisation, factuality |
| Coding | Repository understanding, implementation, debugging, refactoring, code review |
| Mathematics | Exact-answer tasks, derivations, symbolic consistency, proof tasks |
| Reasoning | Multi-step planning, constraint satisfaction, uncertainty calibration |
| Tool use | Correct tool selection, typed arguments, error handling, safe execution |
| Long-horizon work | Progress tracking, recovery, task completion, state preservation |
| Memory | Retrieval relevance, provenance, stale-memory handling, deletion |
| Reliability | Reproducibility, regressions, hallucinated actions, unsupported claims |
| Efficiency | Latency, throughput, accelerator memory, context/cache use and cost |

## 3. Coding evaluation

Use deterministic tasks, sandboxed compile/tests, hidden-test debugging, multi-file repository tasks, build repair, code review and long-running tasks. Execute all candidate code in a disposable, network-disabled environment with resource limits.

## 4. OASIS evaluation

Evaluate the combined model-plus-agent separately: task identification, context quality, tool arguments, permissions, verifier behaviour, retries, final evidence and resume after interruption.

## 5. Regression suite

Maintain a versioned protected suite of English instruction tasks, solved tasks, known failures, privacy/permission boundaries, long-context retrieval, code/build edge cases and tool-failure scenarios.

## 6. Candidate release gates

A candidate must load, match hashes and tokenizer version, show no unexplained instability, pass fixed and held-out tests, avoid unacceptable regressions, pass OASIS permission tests and include a rollback path.

## 7. Reporting

Include model/system versions, code commit, dataset/checkpoint lineage, hardware, evaluator and dataset versions, raw outcomes, failed cases, latency/resources, limitations and a promote/hold/reject recommendation.
