# Compute options for Artemus training

**Last reviewed:** 2026-11-06  
**Goal:** identify realistic hardware paths without promising free capacity or silently incurring costs.

## 1. Free / best-effort GPU paths

### Kaggle Notebooks — first choice for a no-cost pilot
Official notebook documentation reports limited free GPU availability and explains that allocation can be queued or unavailable when resources are busy. The exact free quota and hardware assigned to an individual account can vary; this project does not assume a fixed weekly hour allowance. Kaggle staff guidance says GPU/TPU access is restricted to phone-verified accounts, so complete the Phone Verification section in Kaggle account settings before expecting a GPU worker. An API quota response alone is not proof that a kernel received a GPU: verify CUDA from inside the running notebook.

Official docs: https://www.kaggle.com/docs/notebooks
Account settings: https://www.kaggle.com/settings
Kaggle staff note on phone verification: https://www.kaggle.com/discussions/product-feedback/268333

### Google Colab free tier — useful fallback
Google says free access to GPUs/TPUs is provided on a best-effort basis. Resource availability and usage limits fluctuate and are not published as fixed allowances; GPU types vary. The free managed tier also prohibits remote-control/web-service use and distributed workers, so keep Artemus training inside the interactive notebook workflow. It is not a guaranteed long-running training server.

Official FAQ: https://research.google.com/colaboratory/faq.html

### University / school / hackathon resources
Potentially the best zero-cost route for longer experiments: university research labs, coursework GPU pools, AI clubs, hackathons, open-source research programmes, and cloud credits attached to legitimate student/research programmes. Ask for a time-limited GPU VM and a persistent volume, not indefinite GPU allocation. Do not claim eligibility for a programme until its terms are checked.

### Lightning AI Studios — optional free-credit GPU route
Lightning AI currently advertises a free CPU Studio and conditional introductory GPU credits; its official documentation says GPU-credit eligibility and availability can vary, and that the service is not available in every country. Treat GPU time as a limited trial, not an always-free training server. Check the live pricing and account terms before selecting a GPU; do not enable pay-as-you-go or attach a payment method without an approved budget.

Official pages: https://lightning.ai/docs/platform/build/ai-studio  
Pricing and free-tier details: https://lightning.ai/pricing

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

## 5. GitHub Actions orchestration and Kaggle compute

GitHub Actions is the orchestrator, not the GPU host. The initial launch workflow and recurring Kaggle workflow configure credentials, push the notebook, request the selected accelerator, poll remote status, and verify the exported checkpoint and evaluation artifacts. The actual accelerator is provisioned by Kaggle, not by the GitHub runner. A pushed kernel can still be assigned a CPU environment or fail before training, so the notebook's first cell performs a CUDA/`nvidia-smi` check and exits before any model updates if no usable GPU is visible.

The first automated launch authenticated successfully, showed 30 GPU-hours available in Kaggle's quota response, and uploaded kernel version 1. Kaggle then ran it without a CUDA device, so the foundation loop did not start. This is consistent with account-level GPU eligibility not being enabled; Kaggle staff guidance requires phone verification for GPU/TPU access. Complete phone verification at https://www.kaggle.com/settings, then rerun the initial launch. Do not treat an accepted upload, quota number, or queued kernel as evidence that training occurred.
