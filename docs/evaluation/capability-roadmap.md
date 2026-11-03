# Artemis Flagship Capability Roadmap

**Status (2026-11-03):** Early task-SFT stage. The current model is approximately 1.005M parameters and has been trained for 8,000 steps from random initialization on a small English conversation/math/code mixture. It is still not a coherent assistant. The current continuation workflow resumes verified checkpoints, adds 8,000 steps at a lower cosine learning rate, and runs scripted multi-turn dialogue evaluation. It caps this exact data-mixture run at 16,000 total steps to prevent endless repetition of a small corpus.

## Project invariants

1. Every official Artemis foundation checkpoint begins from random initialization in this repository. External models may be research references or independent evaluators, never the hidden base weights.
2. Licensed data and public research may inform training and architecture. Preserve exact source revisions, licence/attribution obligations, and data hashes.
3. No arbitrary viewpoint or ideology filtering. Data hygiene may target spam, duplication, corruption, privacy/PII, provenance/legal issues, or documented quality defects.
4. Keep generated self-dialogue as evaluation traces unless a human or a separately validated process reviews it. Never blindly train the model on its own unverified outputs.
5. Keep the base model separate from OASIS agent/runtime features. Report model-only and agent-assisted capability separately.
6. No frontier-parity claim without independent, comparable capability results.

## Capability matrix

| Capability | Current coverage | Next work | Required evidence |
|---|---|---|---|
| General language and conversation | OASST1 reviewed English conversation subset | Broaden licensed, diverse prose pretraining; improve instruction data coverage and dialogue quality | Held-out dialogue quality, instruction adherence, factuality and conversation consistency |
| Mathematics and quantitative reasoning | GSM8K train split is in the task mixture | Expand with legally reviewed reasoning/math corpora; keep official test splits protected | Exact-answer accuracy, derivation validity, robustness to paraphrases and multi-step variants |
| Programming and software engineering | MBPP train IDs 601–974 | Add source-licensed code pretraining and verified coding tasks; build a hermetic test runner | Syntax/compile rate, hidden unit tests, pass@k, patch correctness and regression rate |
| Science and technical knowledge | Limited incidental OASST1 coverage | Add curated, attribution-compatible scientific/technical text and original authored question sets | Factual QA, explanation quality, citation/grounding checks and calibrated uncertainty |
| General knowledge and commonsense | Limited conversational coverage | Broad web/encyclopedic pretraining with source-aware holdouts; curated QA tasks | Accuracy by subject, temporal robustness, contradiction handling |
| Bengali | Not yet represented adequately in training | Audit B-CORE component/source obligations; evaluate tokenizer fertility; only ingest after privacy and attribution review | Bengali reading/writing, grammar, QA and summarization on held-out native-speaker-reviewed tasks |
| Banglish and code-switching | Not yet adequately covered | Create original reviewed Banglish examples and evaluate mixed Bengali/English/code snippets | Naturalness, meaning preservation and code-switch robustness |
| Writing and creativity | Small implicit OASST1 coverage | Add original style-controlled writing prompts; measure instruction following | Rubric-based writing quality and constraint adherence |
| Long-context reasoning | Context currently 256 tokens | Increase context only after packed data, attention/cache tests, and a long-context curriculum | Retrieval at multiple positions, multi-document synthesis, exact recall and context stress tests |
| Tool use and agentic work | Not yet trained as a reliable tool caller | Build a separate tool-call schema and verifier-backed traces; integrate through OASIS only after base format is stable | Exact schema validity, tool selection, argument correctness, execution success and recovery |
| Reliability and calibration | Only basic training/validation loss and sample generation | Add abstention/uncertainty examples, adversarially phrased but legitimate questions, and source-grounded checks | Hallucination rate, calibration, consistency, refusal precision and regression tracking |

## Data curriculum

### Stage A — Broad foundation pretraining

The current task-SFT mixture is not a substitute for broad next-token pretraining. A future foundation corpus should mix:
- high-quality general web prose (FineWeb/FineWeb-Edu are candidates, not yet ingested);
- Bengali text (B-CORE is a candidate only after component-level licence/provenance checks);
- technical/scientific prose;
- code repositories and code documentation with source-level licences;
- mathematics and structured technical text.

A corpus manifest must record the dataset revision, source identifiers, licence, attribution, filters, hashes, and train/validation/test separation. Deduplication should be performed at both document and near-duplicate levels, with benchmark contamination checks.

### Stage B — Instruction and capability SFT

The present OASST1/GSM8K/MBPP mixture is an initial task-SFT experiment. Expand it into a balanced curriculum only when each source's rights and quality are checked. Avoid drowning specialist tasks in large conversational text; record sample weights and target-token counts for every domain.

### Stage C — Verifiable tasks

Prefer outcomes that can be checked:
- math: exact final answer and independently checked arithmetic;
- code: parse/compile plus hidden tests in a disposable, network-disabled sandbox;
- tool use: JSON/schema validation and deterministic mock tools;
- factual QA: source-attribution and contradiction checks;
- multilingual: native-speaker reviewed tasks and round-trip/meaning-preservation checks.

Generated outputs must never be executed directly on the runner with broad permissions. Code evaluation must use an explicitly sandboxed, network-disabled process with strict CPU/memory/time limits and no credentials.

### Stage D — Preference and outcome optimisation

Only after Artemis can produce coherent base responses and pass verifiable tasks should we add human preference data, reward models, or outcome-based optimisation. Reward signals need independent verification; an automated model judging its own answers is not sufficient.

## Repeated interaction protocol

Each training cycle should:
1. Resume a checkpoint only after verifying lineage, model/tokenizer compatibility, and exact data hashes (unless a deliberate, recorded data transition is approved).
2. Train for a bounded token/step budget with a recorded learning-rate schedule.
3. Generate multi-turn conversations across the capability matrix.
4. Preserve full transcripts, checkpoint hash, data manifest, training logs, and config.
5. Score only objective checks automatically; keep subjective capability claims out of machine-generated pass/fail summaries.
6. Review regressions before promoting the checkpoint.

The current dialogue harness runs eight two-turn scenarios spanning conversation, math, coding, science, reasoning, Bengali, Banglish, and writing. It records transcripts and basic generation-health signals; it does not claim correctness. The outputs are evaluation material, not automatic training material.

## Scaling and compute gate

The current 1M-parameter CPU model validates engineering mechanics only. The 8.1M, 26.5M and 88.7M candidates are parameter profiles, not trained capable models. A useful scaling study must train matched sizes on the same tokenizer/data recipe and compare equal token or FLOP budgets.

Full frontier ambitions require suitable accelerator compute, broad high-quality token budgets, scalable memory-mapped data loading, checkpoint/resume reliability, and repeatable evaluation. Public GitHub-hosted CPU runners can validate code and run small experiments; they cannot supply frontier pretraining scale.

## Promotion gates

A checkpoint is eligible for promotion only when:
- checkpoint lineage and source licences are auditable;
- data hashes and train/validation/test splits are verified;
- all core CI tests pass;
- it improves or preserves held-out capability performance under matched budgets;
- no severe regression appears in previously passed domains;
- generated conversation examples are not mistaken for verified capability evidence.
