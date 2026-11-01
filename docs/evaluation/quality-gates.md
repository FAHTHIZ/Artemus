# Artemis — Evaluation and Quality Gates

**Status:** Initial evaluation contract.

## 1. Purpose

Evaluation is a release gate, not a marketing layer. A checkpoint or agent change is not an improvement until it demonstrates gains under repeatable conditions without unacceptable regressions.

## 2. Evaluation dimensions

| Dimension | What to test |
|---|---|
| General language | Comprehension, instruction following, summarisation, factuality |
| Coding | Repository understanding, implementation, debugging, refactoring, code review |
| Mathematics | Exact-answer tasks, derivations, symbolic consistency, proof tasks |
| Reasoning | Multi-step planning, constraint satisfaction, uncertainty calibration |
| Tool use | Correct tool selection, typed arguments, error handling, safe execution |
| Long-horizon work | Progress tracking, recovery, task completion, state preservation |
| Memory | Retrieval relevance, provenance, stale-memory handling, deletion |
| Reliability | Reproducibility, regressions, hallucinated actions, unsupported claims |
| Efficiency | Latency, throughput, accelerator memory, context/cache use, cost |

## 3. Coding evaluation

Use several levels:
1. Unit-level coding problems with deterministic tests.
2. Multi-file changes in controlled repositories.
3. Debugging tasks with hidden tests.
4. Build and dependency repair.
5. Repository issue resolution with acceptance criteria.
6. Code review and defect identification.
7. Long-running tasks with checkpoints and changing observations.

Record patch correctness, tests passed, regressions, task success, time, tool calls, retries, and compute. A benchmark score without environment, version, and run instructions is not sufficient.

External public benchmarks can be useful comparisons, but their licence, version, contamination risk, and execution conditions must be checked. Maintain private or newly authored tasks that are not present in training data.

## 4. OASIS evaluation

Evaluate the combined model-plus-agent system separately from the base model:
- Did the planner identify the true task?
- Was context relevant and sufficient?
- Were tools called with valid arguments?
- Did actions stay within permissions?
- Did the verifier detect failures?
- Did retries improve the outcome?
- Was the final report supported by evidence?
- Could the task resume after interruption?

Use deterministic simulated environments first, then controlled real repositories.

## 5. Regression suite

Maintain a versioned protected suite. Include:
- Previously solved tasks.
- Known failure cases.
- Safety and permission boundary tests.
- Bengali and multilingual cases.
- Long-context retrieval tests.
- Code syntax and build edge cases.
- Tool failure and timeout cases.
- False-claim detection (for example, saying a test passed when it did not run).

## 6. Candidate release gates

A candidate must:
- Load and run with the declared configuration.
- Match its artifact hashes and tokenizer version.
- Have no unexplained training instability.
- Be evaluated on fixed and held-out tasks.
- Show no unacceptable regression in core capabilities.
- Pass the OASIS permission and tool-contract suite.
- Have an evaluation report with raw results and known limitations.
- Have a rollback path to the prior stable release.

Thresholds should be set after establishing the first baseline. Do not invent numerical pass thresholds before measuring system variance and task difficulty.

## 7. Reporting format

Each report should include:
- Model and OASIS versions.
- Code commit and configuration.
- Dataset/checkpoint lineage.
- Hardware and software environment.
- Evaluation suite versions.
- Per-task raw results and aggregate metrics.
- Failed cases with error categories.
- Latency and resource use.
- Comparison against the previous stable release.
- Limitations and recommendation: promote, hold, or reject.
