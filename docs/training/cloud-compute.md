# Compute options for Artemus training

**Last reviewed:** 2026-11-06  
**Goal:** identify realistic hardware paths without promising free capacity or silently incurring costs.

## 1. Free / best-effort GPU paths

### Kaggle Notebooks — first choice for a no-cost pilot
Official notebook documentation reports limited free GPU availability (T4 class) and explains that availability/usage are constrained. It is a good first place to run the bundled notebooks/kaggle-gpu-training.ipynb. The exact free quota and hardware assigned to an individual account can vary; this project does not assume a fixed weekly hour allowance.

Official docs: https://www.kaggle.com/docs/notebooks

### Google Colab free tier — useful fallback
Google says free access to GPUs/TPUs is provided on a best-effort basis. Resource availability and usage limits fluctuate; GPU types vary; free notebooks can run for up to 12 hours depending on availability and usage. It is not a guaranteed long-running training server.

Official FAQ: https://research.google.com/colaboratory/faq.html

### University / school / hackathon resources
Potentially the best zero-cost route for longer experiments: university research labs, coursework GPU pools, AI clubs, hackathons, open-source research programmes, and cloud credits attached to legitimate student/research programmes. Ask for a time-limited GPU VM and a persistent volume, not indefinite GPU allocation. Do not claim eligibility for a programme until its terms are checked.

## 2. Paid, on-demand GPU rental

### RunPod
GPU Pods are billed by time used and storage is charged separately. RunPod documents per-second billing and continued storage charges for retained network volumes while Pods are stopped. A Pod may be stopped when the account balance reaches zero; whether data persists depends on the attached volume. Use a persistent volume for checkpoints, set a spending cap outside the training script, and delete unused storage.

Official billing: https://docs.runpod.io/accounts-billing/billing  
GPU Pod pricing: https://docs.runpod.io/pods/pricing

### Vast.ai
Vast is a marketplace with host-dependent prices. It combines compute, storage, and bandwidth charges; storage can continue billing for an instance even while it is stopped/idle depending on instance state. Compare reliability, GPU memory, disk, bandwidth, and host reputation—not just the lowest hourly number. Confirm the current instance billing state before leaving it unused.

Official pricing: https://docs.vast.ai/guides/instances/pricing  
Billing FAQ: https://docs.vast.ai/guides/reference/faq/billing

### Other cloud providers
AWS, Google Cloud, Azure and specialist GPU providers can be useful with research or startup credits, but GPU VM, storage, network egress, and snapshot costs are separate. No cloud resource has been provisioned for this repository. Do not start a paid instance until a total budget, shutdown plan, and checkpoint destination are clear.

## 3. A sensible compute progression

| Step | Hardware | Work |
|---|---|---|
| A | GitHub Actions CPU | Unit tests, data integrity, tiny smoke training only |
| B | Kaggle / Colab GPU when available | 1M–10M parameter experiments, tokenizer and loader checks, small corpus pilots |
| C | Single paid A10/A100/L40-class GPU or institutional GPU | Larger model and longer pretraining; test BF16/FP16, context and throughput |
| D | Multi-GPU server | Only after a single-GPU baseline and a measured scaling need exist |
| E | Multi-node cluster | Far future; communication cost and engineering overhead will exceed the value of an early experiment |

Artemus is currently around one million parameters in its compact experimental checkpoint. A single GPU is sufficient to improve the data/training pipeline and run controlled small-model experiments. Multi-node compute is not the next bottleneck.

## 4. Cloud training checklist

Before launching:
- verify GPU model, VRAM, driver and CUDA version;
- estimate model, optimizer, activation and KV-cache memory;
- use a pinned dataset commit and a local/persistent cache;
- checkpoint every few hundred steps and upload provenance alongside weights;
- use mixed precision only when the GPU supports it; T4-class devices generally use FP16, while supported A100-class devices can use BF16;
- use gradient accumulation to fit an effective batch without exceeding VRAM;
- record actual tokens processed, wall time and tokens/second;
- shut down compute immediately after validation/export;
- verify the saved checkpoint can be loaded from persistent storage before deleting the instance.

## 5. Why GitHub Actions is not the training cloud

The current Actions training workflow is a CPU runner used for reproducible pipeline checks and small experiments. It is not a general GPU pool. The repository now has a Kaggle notebook for user-controlled GPU training; this notebook can be run when the account has a GPU allocation. It does not run automatically from a Git push and it cannot provision a cloud account on the user's behalf.
