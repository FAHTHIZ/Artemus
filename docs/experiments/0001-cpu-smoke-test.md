# Experiment 0001 — CPU training-pipeline smoke test

**Date:** 2026-11-02  
**Status:** Completed; pipeline validation only  
**Model lineage:** Random initialization; no imported pretrained weights

## Purpose

Verify that the initial training code can instantiate a model from random weights, train on a small UTF-8 corpus, calculate validation loss, save a checkpoint, reload it, and generate a sample.

This is not a foundation-model pretraining run in the sense required to produce a useful general-purpose assistant. The corpus is a small synthetic demonstration file and is not suitable as the substantive Artemis training corpus.

## Environment

| Item | Value |
|---|---|
| Runtime | Linux x86_64 |
| Python | 3.13.5 |
| PyTorch | 2.10.0+cpu |
| Accelerator | CPU only; CUDA unavailable |
| CPU threads | 3 |
| Model parameters | 119,808 |
| Tokenizer | UTF-8 byte tokenizer, 257 token IDs |
| Context length | 64 tokens |
| Transformer layers | 2 |
| Attention heads | 2 |
| Embedding width | 64 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Training steps | 250 |
| Seed | 1337 |

## Dataset

- File: `examples/demo_corpus.txt`
- Training split: 2,821 byte tokens
- Validation split: 314 byte tokens
- Content: short synthetic explanations of language modelling, software engineering, debugging, testing, and Bengali text.

The split is a simple sequential 90/10 split. The validation set is too small and too closely related to the training material to support broad generalization claims.

## Results

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 5.5049 | 5.5286 |
| 50 | 4.1032 | 4.0293 |
| 100 | 3.2783 | 3.3257 |
| 150 | 2.9269 | 3.0475 |
| 200 | 2.7078 | 2.9064 |
| 250 | 2.5417 | 2.7942 |

The falling loss indicates that the model learned patterns in this small demonstration corpus. It does not show that the model has meaningful general language or coding capability. A generated sample was mostly incoherent, as expected at this size and training-data scale.

## Verification

- Unit tests on the local training copy: **8 passed**.
- Training: completed for 250 steps with finite loss.
- Checkpoint: saved successfully (approximately 1.4 MiB).
- Inference: checkpoint reloaded and generated 80 additional tokens.
- Resume test: loaded the step-250 checkpoint, restored model/optimizer state, and completed five further steps to step 255; validation loss was 2.7912.
- Checkpoint SHA-256 is recorded in the accompanying local run artifact.

## Important implementation note

Review during the run identified that the token embedding and output projection shared a weight before recursive initialization, causing the shared matrix to be initialized twice. The repository baseline has been corrected so weights are initialized first and tied afterward. The local run used this corrected initialization order.

## Artifacts

The checkpoint and full local run package are intentionally kept outside Git because checkpoints are generated artifacts. The checkpoint and run package are supplied separately for reproducibility of this smoke test.

## Conclusion

The project has completed its first verified CPU training-loop test. It has **not** trained a useful Artemis assistant. The next substantive work is hardware discovery, a properly sourced and licensed data pipeline, an evaluation corpus independent of training data, and a training configuration sized to available compute.
