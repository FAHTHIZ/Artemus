# Experiment 0007 — Multi-task CPU task-SFT run from random initialization

**Date:** 2026-11-02  
**Status:** Completed successfully as a training-pipeline run; generated answers are still incoherent  
**Workflow:** https://github.com/FAHTHIZ/Artemus/actions/runs/41372200736  
**Training source branch/commit:** `bootstrap/architecture` / `4d239e9640024a40803b44c95dd5a116f9c7aaa5`  
**Artifact:** `artemis-capability-mix-cpu-run`, GitHub Actions artifact ID `13109929242`  
**Artifact expiry:** 2026-11-09  
**Artifact ZIP SHA-256:** `e5bec0df7d454e84b6b65a79ed13f7f323a586d42b0239bbe193f0bbed2ef2b9`

## Objective

Test whether a small random-init model can train over a mixed set of conversational, mathematical, and Python-programming tasks using response-only supervision. This is an initial multi-task supervised experiment. It is **not** broad next-token foundation pretraining and must not be described as such.

## Data sources and integrity

### OASST1

- Source revision: `a4992e1595393714c9e6c567b1ed645bbf7d00ee`
- Licence: Apache-2.0
- Source SHA-256: `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c`
- Filters: English, reviewed, not deleted, non-synthetic, positive PII-label examples excluded.
- 39,630 eligible messages reconstructed into 23,933 conversation examples.
- 21,402 training examples / 2,531 validation examples.
- Conversation trees were grouped by `message_tree_id`; no tree crosses splits.

### GSM8K

- Source commit: `3101c7d5072418e28b9008a6636bde82a006892c`
- Licence: MIT
- Source SHA-256: `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`
- Official training set only; 7,473 examples split deterministically by question/answer hash into 6,701 training / 772 validation.
- Official test split was not downloaded or ingested.

### MBPP

- Source commit: `83f1e7b59e762648c953de751d31f75720683622`
- Dataset card licence: CC-BY-4.0; attribution obligations apply.
- Pinned Git blob SHA: `96e2dc2f7dd95b7b87b38dc46e42cbed444251bb`
- Official task IDs 601–974: 374 training examples.
- Official task IDs 511–600: 90 validation examples.
- IDs 1–10 (prompt) and 11–510 (test) excluded.

### Combined mixture

| Split/source | Training | Validation |
|---|---:|---:|
| OASST1 conversations | 21,402 examples | 2,531 |
| GSM8K | 6,701 examples | 772 |
| MBPP | 374 tasks | 90 |
| SentencePiece tokens | 10,270,249 | 1,172,397 |
| Assistant response target tokens | 8,255,369 | 936,664 |

The preparation manifest SHA-256 is `b048dc6f26924913e9e31828b9617914de68bb37ee79ed900c8603b6ae897776`. The OASST1 manifest SHA-256 is `9a32eace8e04af1b0e2b222008c30e146a3e387710f97ad66b7f4a296ea107f1`.

## Training configuration

| Setting | Value |
|---|---:|
| Initialization | Random; no pretrained checkpoint |
| Parameters | 1,347,200 |
| Architecture | 4-layer dense Transformer |
| Attention | Full MHA (4 query / 4 KV heads) |
| Width / heads | 128 / 4 |
| Position | Learned absolute embeddings |
| Feed-forward | GELU |
| Tokenizer | SentencePiece BPE with byte fallback |
| Vocabulary | 4,096 |
| Context | 256 tokens |
| Objective | Assistant response and end-of-turn targets only |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Batch size | 8 |
| Steps | 8,000 |
| Evaluation interval | 250 |
| Evaluation batches | 8 |
| Seed | 1337 |
| Device | GitHub-hosted CPU runner |

This run predates the factorized-embedding configuration and the compact GQA/RoPE/SwiGLU candidate; it used the legacy full-MHA/learned-position/GELU architecture.

## Loss curve

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 8.2767 | 8.2641 |
| 1,000 | 5.5185 | 5.5045 |
| 2,000 | 5.1481 | 5.1062 |
| 3,000 | 4.8160 | 4.9203 |
| 4,000 | 4.4683 | 4.7354 |
| 5,000 | 4.4971 | 4.6618 |
| 6,000 | 4.1547 | 4.4724 |
| 7,000 | 4.1133 | 4.3236 |
| 8,000 | **4.0091** | **4.3163** |

Loss decreased substantially, but this measures the response-only training objective and is not a benchmark score. The training loop samples windows from the same finite corpus; the number of sampled token positions is not the number of unique fresh tokens.

## Generated samples

### Mathematics prompt

Prompt: “A shop sells 3 pens for $6. What is the price of 5 pens at the same rate?”

Generated text began:

```text
'$ ($)$) = 4. The speed of each length of 10 is $24 = 8 - 3 = 4 = 8q$7 ...
```

### Python prompt

Prompt: “Write a Python function that adds two numbers.”

Generated text was malformed, with incomplete punctuation, symbols, and no valid function.

The generation step succeeded technically; both answers failed the quality expectation. The checkpoint is not a usable assistant.

## Verification

- Unit and data-integrity suite: **21 passed**.
- OASST1 download/checksum and grouped split: passed.
- GSM8K download/SHA-256: passed.
- MBPP download/Git blob SHA: passed.
- Mixture creation: passed.
- 8,000 optimizer steps: completed.
- Checkpoint reload/generation: passed.
- Math and Python sample generation: ran, but outputs were incoherent.

## Checkpoint and provenance limitation

- Checkpoint SHA-256: `46c550b303c971fffd2da7ab1204827ced595558a69259297f47226db4d735b1`
- Checkpoint format version: **2**
- The checkpoint has no embedded `run_metadata`; the original training workflow did not pass a data-manifest or phase argument.
- The source commit, workflow run, command line, source manifests, training logs, and artifact hashes above provide a sidecar lineage record. This is a provenance limitation; it must not be hidden.
- The updated manual workflow on `main` now passes `--data-manifest` and `--phase task_sft` and emits checkpoint format v3 with run metadata.

## Assessment and next step

This run verifies the data path and 8,000-step CPU training mechanics on a richer mixture. It does not establish mathematical ability, coding ability, general reasoning, or frontier performance. The output demonstrates that the model remains under-capacity and undertrained for coherent generation.

Next:
1. Make the new checkpoint-v3 provenance path the only path for new runs.
2. Add automatic held-out GSM8K/MBPP evaluation with protected test splits and sandboxed validators.
3. Compare baseline and factorized/GQA variants under identical tokenizer, dataset, seed, token budget, and evaluation.
4. Start true foundation pretraining only when a broad licensed corpus and real accelerator budget are available.
