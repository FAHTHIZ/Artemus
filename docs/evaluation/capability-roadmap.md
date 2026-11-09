# Artemis — English capability roadmap

## 1. Capability priorities

| Domain | Initial training route | Evaluation |
|---|---|---|
| English conversation and instruction following | Reviewed OASST1; Aya `eng` only | Multi-turn instruction adherence, context retention, factuality |
| Mathematics | GSM8K train subset; verified project tasks | Exact answers, multi-step consistency and protected holdouts |
| Python and software engineering | Approved MBPP tasks; OSCR files with permissive source-level licences | Syntax, compile, hidden tests, repository changes in isolated environments |
| Science and technical knowledge | English FineWeb / FineWeb-Edu; reviewed specialist sources | Comprehension, grounded explanation and held-out synthesis |
| Reasoning and planning | English project-authored tasks and independently checked outcomes | Constraint satisfaction, paraphrase robustness and multi-step correctness |
| Tool use | OASIS adapters and deterministic simulated environments | Correct actions, permission boundaries, recovery and independent verification |
| Long-horizon work | Checkpointed project tasks | Completion, state retention, recovery and evidence-backed reporting |

## 2. Active data boundaries

Foundation text must pass the English label and ≥0.70 language-confidence gate. Aya examples must have `language_code=eng`. OASST1 messages must pass its reviewed-English filter. Math and code datasets use only approved training splits. No non-English corpus, text sample, tokenizer target or dialogue scenario may enter the active pipeline.

## 3. Evaluation integrity

- Pin releases, task IDs, prompts and evaluator code.
- Keep protected tests out of training, tokenizer fitting and prompt examples.
- Maintain private or newly authored holdouts to reduce contamination.
- Execute generated code only in a disposable, resource-limited, network-disabled sandbox.
- Report per-task outcomes, uncertainty, regressions and failures.
- Evaluate ArtemisLM separately from ArtemisLM + OASIS.

## 4. Dialogue health

The dialogue harness uses English-only two-turn scenarios for conversation, mathematics, programming, science, reasoning, reading comprehension, planning and writing. It measures generation health and context handling; it does not claim correctness by itself.

## 5. Promotion

A candidate requires verified checkpoint lineage, source licences, data hashes, passing CI, held-out capability results, no unacceptable core regressions, a rollback path and a written evaluation report. A successful training run alone does not certify a useful assistant.
