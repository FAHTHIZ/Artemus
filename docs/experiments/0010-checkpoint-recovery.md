# Experiment 0010 — Recovered 16k task-SFT checkpoint and dialogue evaluation

**Date:** 2026-11-06  
**Workflow:** [Run 42233796727](https://github.com/FAHTHIZ/Artemus/actions/runs/42233796727)  
**Result:** Workflow succeeded; reused existing checkpoint; no duplicate training run was performed.

## Why this run exists

A previous continuation run reached 16,000 optimizer steps and saved a format-v3 checkpoint, but its final dialogue evaluation failed because one English follow-up prompt exceeded the model's 256-token context window. The checkpoint artifact was uploaded despite that post-training failure.

The continuation workflow was changed to select the newest completed run that contains the expected non-expired checkpoint artifact, rather than requiring every post-training evaluation step to have succeeded. The dialogue evaluator now trims the oldest complete user/assistant turns when the context would overflow. It never silently truncates the current user prompt.

## Recovery outcome

- Checkpoint format: v3.
- Checkpoint step: 16,000.
- Parameters: 1,004,928.
- The workflow verified checkpoint lineage and the existing data mixture.
- The same mixture had already reached its configured 16,000-step budget, so the training step was intentionally skipped.
- The model was loaded and evaluated from that checkpoint; the artifact was re-uploaded with a new evaluation trace.

## Dialogue health result

| Metric | Result |
|---|---:|
| Scenarios | 8 |
| Turns | 16 |
| Non-empty responses | 16 |
| Empty responses | 0 |
| Context-trimmed turns | 1 |
| Old history messages dropped | 2 |
| Context overflows | 0 |

This is a dialogue pipeline and generation-health check, not a correctness benchmark. The run explicitly notes that generated text is not automatically added to training.

## Capability spot checks

The math prompt produced nonsensical arithmetic text. The Python prompt produced repetitive, incoherent text. Those results confirm that successful loss reduction, checkpoint recovery, and non-empty dialogue generation do not establish useful reasoning or coding ability.

## Decision

Do not repeat the OASST1/GSM8K/MBPP task mixture beyond its current budget. Keep this checkpoint as a historical task-SFT baseline. Start a separate foundation-pretraining run from random initialization using a pinned, broader corpus and a newly trained tokenizer. Evaluate that checkpoint against the same protected and private tests, then add code, mathematics and reviewed human-teaching data as separate stages.

The new general pretraining entry point is `notebooks/kaggle-gpu-training.ipynb`; actual GPU allocation requires a Kaggle account with an available GPU accelerator.
