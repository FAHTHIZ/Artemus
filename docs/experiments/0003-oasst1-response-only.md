# Experiment 0003 — OASST1 response-only supervision

**Date:** 2026-11-02  
**Status:** Completed; tests, training, checkpoint reload, and generation passed; assistant quality remains inadequate  
**Workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41371151466  
**Artifact:** `artemis-oasst1-response-only-run` (seven-day GitHub Actions retention)

## Objective

Test whether supervising only assistant reply tokens while retaining the user prompt as context improves the next-token training objective for a small random-init Artemis model.

## Dataset

Pinned OASST1 prepared data:
- 15,000 reconstructed conversation examples.
- 13,393 training examples; 1,607 validation examples.
- Conversation trees held together across the split.
- 6,419,439 training SentencePiece tokens; 724,275 validation tokens.
- 5,475,663 assistant-target tokens in training and 614,878 in validation.

## Configuration

| Setting | Value |
|---|---|
| Initialization | Random; no imported pretrained weights |
| Parameters | 1,085,056 |
| Layers / heads / width | 4 / 4 / 128 |
| Tokenizer | SentencePiece BPE with byte fallback |
| Vocabulary | 2,048 |
| Context | 256 tokens |
| Loss | Assistant reply and end-of-turn targets only |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Batch size | 8 |
| Steps | 1,200 |
| Evaluation interval | 100 |
| Seed | 1337 |
| Runtime | GitHub-hosted CPU |

## Loss curve

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 7.5959 | 7.5980 |
| 100 | 6.4750 | 6.5448 |
| 200 | 6.0796 | 6.1067 |
| 300 | 5.7765 | 5.8369 |
| 400 | 5.6188 | 5.6694 |
| 500 | 5.4254 | 5.5168 |
| 600 | 5.3429 | 5.3434 |
| 700 | 5.2698 | 5.2647 |
| 800 | 5.1121 | 5.1615 |
| 900 | 5.0872 | 5.0447 |
| 1,000 | 4.9603 | 5.0213 |
| 1,100 | 4.9065 | 5.0341 |
| 1,200 | 4.8339 | 4.9028 |

These are response-only objective values and must not be compared directly with full-token loss from Experiment 0002.

## Verification

- Unit tests: **17 passed**.
- Dataset hash and preprocessing: passed.
- Conversation grouping: passed.
- Response-only mask: tested for user text, assistant text, and end markers.
- Training: 1,200 steps completed with finite losses.
- Checkpoint save/reload and generation: passed.

## Quality assessment

Generated sample for “What is a variable?”:

```text
Some of AI is the LEIP. AM, the TEUITRGL
```

This is not a coherent answer. Response-only masking is a valid mechanics improvement, but it did not turn the 1.085M-parameter model into a useful assistant.

The run is an SFT-style pipeline experiment from random weights, not broad language-model foundation pretraining. Its limitations include tiny model capacity, low effective training compute, limited/old dialogue data, and no broad next-token pretraining phase. Do not promote this checkpoint as a chat assistant.

Preserve the checkpoint hash and run logs from the linked Actions artifact before retention expires. Binary checkpoints remain excluded from Git.
