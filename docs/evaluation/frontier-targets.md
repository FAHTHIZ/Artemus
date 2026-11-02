# Frontier Comparison and Evaluation Plan

**Purpose:** Measure progress toward frontier-class capability without overstating early results.

## Reference systems

As of 2026-11-02, official provider pages identify:
- OpenAI GPT-6 Astra as its highest-capability general model and GPT-6.1 Sol as a lower-cost model described as near-Astra on complex work.
- Anthropic Claude Fable 5.1 as a generally available high-capability model for coding and knowledge work.

Official references:
- OpenAI model catalogue: https://developers.openai.com/api/docs/models
- GPT-6.1 Sol: https://openai.com/index/introducing-gpt-6-1-sol/
- Claude Fable 5.1: https://www.anthropic.com/claude/fable
- Fable/Mythos 5.1 announcement: https://www.anthropic.com/claude-fable-and-mythos-5-1

Provider-published results use provider-specific harnesses and settings. Do not compare them directly with Artemis scores unless benchmark version, split, prompting, tools, sampling, and scoring protocol align. Prefer independent evaluation.

## Capability scorecard

Report separate results for general knowledge/comprehension; instruction following and conversation; mathematics; code generation and unit-test pass rate; repository-level software engineering; debugging/build repair/code review; long-horizon tool use; Bengali/Banglish/multilingual understanding; factuality/calibration/refusal correctness; and memory/context retrieval.

## Base model vs system

Every report separates:
- **Artemis base model:** no retrieval, no external tools, fixed context and decoding.
- **Artemis + OASIS:** tools, retrieval, memory, execution, and verification enabled.

Tool-assisted success must not be mistaken for intrinsic model ability. Record tools and permissions.

## Coding evaluation

Use layered tasks: deterministic Python problems with hidden tests; multi-file repositories; bugs with regression tests; repository issues with acceptance criteria; and code review cases. Run candidate code only in isolated, network-disabled sandboxes.

Record patches, test outcomes, failures, tool traces, latency, token budget, and compute.

## Protocol

- Pin benchmark release and commit.
- Keep benchmark test sets out of tokenizer and training.
- Use equivalent system prompts and decoding parameters for direct comparisons.
- Make tool availability identical for agent comparisons.
- Run multiple trials where randomness matters.
- Publish per-task outcomes and uncertainty.
- Keep private/newly authored holdout tasks to reduce contamination.
- Record known contamination and version changes.

## Promotion standard

An Artemis release may claim improvement only when it beats the previous Artemis checkpoint under the same protocol and reports regressions. Claim frontier parity only for a named capability and benchmark with independent, comparable results. Broad frontier parity requires evidence across the full scorecard.
