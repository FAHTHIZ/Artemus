# Frontier Comparison and Evaluation Plan

**Purpose:** define evidence required to call Artemis a compact frontier flagship.

## Reference systems

Use current official provider documentation for GPT, Claude, DeepSeek and other reference systems. Provider-published results are useful context, not directly comparable guarantees: prompts, tools, reasoning budgets, inference settings, test splits and scoring pipelines differ.

Official references:
- https://developers.openai.com/api/docs/models
- https://www.anthropic.com/claude
- https://www.deepseek.com/
- https://huggingface.co/allenai/Olmo-3-1025-7B

## English-first flagship scorecard

1. Software engineering: synthesis, debugging, code review, test repair, APIs and repository-level work.
2. Reasoning: multi-step logic, planning, causal analysis and paraphrase robustness.
3. Mathematics: arithmetic, algebra, geometry, proof tasks and exact-verifier word problems.
4. General knowledge: English comprehension, factual retrieval, summarization and uncertainty calibration.
5. Tool use: shell, Git, Python, browser/search and structured APIs.
6. Long-horizon reliability: task completion, recovery and state retention.
7. Interaction quality: English instruction following, coherent dialogue and style control.
8. Trustworthiness: factuality, citation correctness, privacy handling and calibration.
9. Efficiency: parameters, training tokens/FLOPs, inference throughput, memory, latency, energy and cost per solved task.

## Base model versus agent system

Publish separate results for ArtemisLM without retrieval/tools and for ArtemisLM + OASIS with external tools, memory, planning, execution and verification. Never attribute OASIS or external-tool success to raw model weights.

## Benchmark integrity

Pin releases, preserve test splits, deduplicate against training, maintain private holdouts, report raw trials and match inference budgets. Public benchmarks support iteration but do not replace fresh project-authored tasks.

## Promotion rules

A capability claim requires strong results in that domain. A frontier-parity claim requires independent, comparable results across this scorecard with competitive latency/cost. One benchmark, one demonstration or one fluent response is not evidence of broad parity.
