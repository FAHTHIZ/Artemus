# Frontier Comparison and Evaluation Plan

**Research snapshot:** 2026-11-02  
**Purpose:** Define the evidence required to call Artemis a compact frontier flagship.

## Reference systems

Current official provider pages identify:
- OpenAI **GPT-6 Astra** as its flagship model for demanding reasoning and coding.
- OpenAI **GPT-6.1 Sol** as a lower-cost system described by its provider as near-Astra on complex work.
- Anthropic **Claude Fable 5.1** as a high-capability model for coding and knowledge work.

Official references:
- https://developers.openai.com/api/docs/models
- https://developers.openai.com/api/docs/guides/latest-model
- https://openai.com/index/introducing-gpt-6-1-sol/
- https://www.anthropic.com/claude/fable
- https://www.anthropic.com/claude-fable-and-mythos-5-1

Provider-published scores are useful reference points, not directly comparable guarantees. Different prompts, tools, reasoning budgets, fallback models, test splits, and scoring pipelines can change results. Artemis must record those details and prefer third-party or independently implemented evaluation.

## The flagship scorecard

No single aggregate score is sufficient. Publish per-capability results for:

1. **Software engineering:** function-level code synthesis, debugging, code review, test repair, API integration, repository-level feature work, and long-horizon multi-file tasks.
2. **Reasoning:** multi-step logic, planning, causal analysis, ambiguity handling, and robust performance under paraphrase.
3. **Mathematics:** arithmetic, algebra, geometry, proofs, numerical methods, and word problems with exact verifiers.
4. **General knowledge:** comprehension, factual retrieval, summarization, document/table understanding, and calibrated uncertainty.
5. **Multilingual ability:** English, standard Bangla, colloquial Bangla, Banglish, and a broader language set selected from actual data coverage.
6. **Tool use:** shell, Git, Python, browser/search, structured APIs, and recovery from tool failures.
7. **Long-horizon reliability:** task completion across many steps, state retention, recovery, and correct final verification.
8. **Interaction quality:** instruction following, coherent dialogue, style control, and useful clarification behaviour.
9. **Trustworthiness:** factuality, citation correctness, privacy handling, refusal precision, and calibration.
10. **Efficiency:** total/active parameters, training tokens/FLOPs, inference tokens/s, peak memory, KV-cache bytes, latency, energy, and cost per solved task.

## Base model versus agent system

Every report must publish two distinct scores:
- **Artemis base model:** no retrieval or external tools; fixed prompt, context, sampling, and token budget.
- **Artemis + OASIS:** tool use, retrieval, memory, planning, code execution, and verification enabled.

Never attribute OASIS or external-tool success to raw model weights. Run matched model-plus-agent comparisons with identical tool permissions and comparable compute budgets.

## Coding evaluation

Use several layers:
- Deterministic algorithm problems with private hidden tests.
- Python and other language tasks with compile/test execution in isolated sandboxes.
- Multi-file repository tasks with explicit acceptance criteria.
- Debugging and regression repair tasks.
- Freshly authored, private issue-style tasks that have not appeared in public training corpora.

MBPP and similar small public sets are useful for development, not sufficient for flagship certification. OpenAI reported contamination in SWE-bench Verified and recommends SWE-bench Pro instead; use current benchmark versions and still maintain private holdouts. Reference: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/

For all execution tests, use a disposable, network-disabled sandbox with CPU, memory, filesystem, process, and wall-time limits.

## Benchmark integrity

- Pin dataset/benchmark releases, commits, task IDs, prompts, and evaluator code.
- Keep test splits out of training, tokenizer fitting, prompt examples, and synthetic task generation.
- Deduplicate across pretraining, SFT, and evaluation data; record likely contamination.
- Keep a private holdout set and rotate it carefully.
- Use multiple trials where generation is stochastic; publish confidence intervals or at least the raw trial outcomes.
- Report scores at matched reasoning tokens, context, tools, retries, and inference compute.
- Publish failures and regressions, not only solved examples.

## Promotion rules

An internal improvement claim requires a reproducible win against the prior Artemis checkpoint under the same protocol. A specialist capability claim requires strong task-level results in that domain. A frontier-parity claim requires independent, comparable performance across the whole relevant scorecard, plus competitive latency/cost under a clearly stated operating budget. Broad flagship parity cannot be inferred from one benchmark or one model-assisted demo.
