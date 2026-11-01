# First Training Run — Tiny Baseline

This guide runs the initial randomly initialized byte-level Transformer. It is a pipeline smoke test only; the byte tokenizer and small model are not the planned final architecture.

## Requirements

- Python 3.10 or newer.
- PyTorch 2.3 or newer.
- A UTF-8 plain-text file with enough data for both training and validation splits.

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Install a PyTorch build appropriate for the local hardware when necessary. The default `--device auto` selects CUDA, then Apple MPS, then CPU.

## Run the smoke test

```bash
artemus-train \
  --data examples/demo_corpus.txt \
  --steps 20 \
  --batch-size 4 \
  --block-size 32 \
  --layers 2 \
  --heads 2 \
  --embedding-size 32 \
  --eval-interval 10 \
  --output artifacts/artemis-tiny.pt
```

The generated checkpoint is local and ignored by Git. Training data and checkpoints must not be committed.

To continue a run from a checkpoint, use `--resume PATH` and set `--steps` to the new total target step count (not the number of additional steps).

## What this verifies

- UTF-8 bytes can be tokenized and decoded.
- The model starts with randomly initialized parameters.
- Attention is causal: changing future tokens must not change earlier-position logits.
- A training step computes finite loss and updates parameters.
- Evaluation, checkpoint save, and checkpoint resume work.

## What it does not prove

This demonstration does not establish useful general intelligence, high-quality coding, production tokenizer quality, long-context capability, distributed training, frontier performance, or a final architecture choice. The demo corpus is only for smoke testing. Serious training requires a properly sourced and licensed corpus, held-out evaluations, and compute planning based on actual hardware.
