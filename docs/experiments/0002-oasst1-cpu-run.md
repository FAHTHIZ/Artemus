# Experiment 0002 — OASST1 conversation-data CPU run

**Date:** 2026-11-02  
**Status:** Completed successfully; training pipeline and checkpoint verified; conversational quality inadequate  
**Branch:** `bootstrap/architecture`  
**Run:** https://github.com/FAHTHIZ/Artemus/actions/runs/41369624968  
**Artifact:** `artemis-oasst1-cpu-run` (retained by GitHub for seven days)

## Objective

Move beyond the synthetic smoke-test corpus and verify that Artemis can ingest a pinned, licensed conversation dataset, preserve a held-out validation split, train from random initialization, save a checkpoint, reload it, and generate a response.

This is still a small pipeline experiment, not an attempt to claim a useful general-purpose assistant.

## Source and provenance

- Dataset: OpenAssistant Conversations (OASST1), ready-for-export message archive.
- Dataset card: https://huggingface.co/datasets/OpenAssistant/oasst1
- Dataset-card licence: Apache-2.0.
- Immutable source revision: `a4992e1595393714c9e6c567b1ed645bbf7d00ee`.
- Archive: `2023-04-12_oasst_ready.messages.jsonl.gz`.
- Archive SHA-256: `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c`.
- Downloaded archive size: approximately 32.61 MiB.
- The SHA-256 verification passed before preprocessing.

The preparation script retained English messages only, required `review_result=true` and `deleted=false`, excluded synthetic records and records with a positive PII label, reconstructed parent chains, and kept conversation trees together when splitting.

## Dataset preparation results

| Metric | Result |
|---|---:|
| Messages remaining after filters | 39,630 |
| Conversation examples selected | 15,000 |
| Training examples | 13,393 |
| Validation examples | 1,607 |
| Training conversation-tree groups | 3,233 |
| Validation conversation-tree groups | 385 |
| Training byte tokens | 19,954,349 |
| Validation byte tokens | 2,246,124 |

The split is deterministic by a hash of `message_tree_id`; no conversation tree crosses from training into validation. The raw archive and processed text were not committed to Git.

## Training configuration

| Setting | Value |
|---|---|
| Initialization | Random; no imported pretrained weights |
| Parameters | 119,808 |
| Architecture | Decoder-only Transformer |
| Layers / attention heads | 2 / 2 |
| Embedding width | 64 |
| Context length | 64 UTF-8 bytes |
| Tokenizer | UTF-8 byte tokenizer, 257 IDs |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Batch size | 8 |
| Training steps | 1,200 |
| Evaluation interval | 100 steps |
| Evaluation batches per split | 8 |
| Seed | 1337 |
| Runtime | GitHub-hosted Linux CPU runner |
| Python / PyTorch | Python 3.11 / CPU PyTorch |

## Loss curve

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 5.5044 | 5.4979 |
| 100 | 3.3708 | 3.3309 |
| 200 | 2.9733 | 2.9247 |
| 300 | 2.8155 | 2.8653 |
| 400 | 2.6842 | 2.7984 |
| 500 | 2.7017 | 2.7414 |
| 600 | 2.6750 | 2.6583 |
| 700 | 2.6152 | 2.6359 |
| 800 | 2.6647 | 2.6416 |
| 900 | 2.5750 | 2.7222 |
| 1,000 | 2.5728 | 2.5182 |
| 1,100 | 2.5846 | 2.5704 |
| 1,200 | 2.5581 | 2.5500 |

Loss reduction confirms that optimisation learned statistical patterns in the corpus. It does **not** establish that the model learned useful dialogue, reasoning, or coding.

## Verification

- Unit tests: **11 passed**.
- Dataset SHA-256 check: **passed**.
- Preprocessing: **completed**.
- Training: **1,200 steps completed**; all reported losses finite.
- Checkpoint save and reload: **passed**.
- Text generation: **executed**.
- Artifact upload: **completed**; checkpoint and preprocessing manifest included.

## Quality assessment

**The generated text was incoherent and did not answer the prompts.** The checkpoint is not suitable for deployment or for use as a chat assistant.

The likely bottlenecks are structural and data/sequence related:

1. A byte-level tokenizer spends many tokens representing ordinary text; the 64-byte context is extremely short in natural-language terms.
2. The model has only 119,808 parameters, far too little capacity for the intended conversational task.
3. The current trainer is plain next-token prediction, not supervised instruction tuning with response masking, preference optimisation, or verified task outcomes.
4. The dataset is useful as a first conversation source, but the selected 15,000 examples are only a small subset of the available archive.
5. Lower token loss is not a substitute for held-out task-based quality evaluation.

Do not simply repeat this run for more steps or call the output an assistant. The next experiment should first improve tokenisation and context length, add explicit instruction-response evaluation, and then measure whether a somewhat larger model can learn coherent continuations on the same pinned corpus.

## Reproducibility and artifact

- Workflow source: `.github/workflows/validation.yml`
- Dataset builder: `src/artemis/prepare_oasst1.py`
- Dataset manifest: `artifacts/oasst1/manifest.json` inside the run artifact.
- Checkpoint: `artifacts/artemis-oasst1.pt` inside the run artifact.
- GitHub Actions run: https://github.com/FAHTHIZ/Artemus/actions/runs/41369624968
- Artifact URL: https://github.com/FAHTHIZ/Artemus/actions/runs/41369624968/artifacts/13108513170

GitHub Actions artifacts expire after seven days in this run configuration. Preserve the checkpoint separately before that date if it is needed later.
