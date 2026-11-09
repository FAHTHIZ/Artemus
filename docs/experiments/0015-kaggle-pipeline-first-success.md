# Experiment 0015: first successful end-to-end Kaggle training cycle

**Date:** 2026-11-09
**Status:** Pipeline verified. GPU not yet allocated (account-side)
**Run:** GitHub Actions [run 42616183342](https://github.com/FAHTHIZ/Artemus/actions/runs/42616183342), commit `aac98fd`

## What was fixed (launch attempts 1–4)

| Attempt | Failure | Fix |
|---|---|---|
| 1 (`9139fc3`) | `artemus-source.zip` missing on the mounted input | Kaggle extracts uploaded `.zip` files, so the bundle archives are now named `.zipdata` and unpacked with `format='zip'` (`acbcc25`) |
| 2 (`acbcc25`) | Bundle verification wanted `dataset-metadata.json` | Kaggle consumes that file as dataset settings; it is no longer listed in `bundle-manifest.json` (`73ebe68`) |
| 3 (`73ebe68`) | `ModuleNotFoundError: artemis` | The editable install's `.pth` is only read by new interpreters; the notebook now prepends `<workdir>/src` to `sys.path` (`aac98fd`) |
| 4 (`aac98fd`) | none: **success** | |

The workflow also now waits until Kaggle has processed the input dataset version for the exact commit before pushing the kernel.

## Result of attempt 4

- Kaggle reported `cuda_available: false`, so the notebook used its bounded CPU smoke profile: 1,267,072 params, foundation 1,000 steps, Aya SFT 250, task SFT 250, batch 1.
- Ran: foundation → Aya SFT → task SFT, capability eval, dialogue eval, export of all 3 checkpoints, and creation of the private `artemus-training-state` Kaggle dataset. The weekly `continuous-kaggle-training.yml` cycle resumes from that dataset.
- Results are uploaded as the run artefact `artemus-initial-training-42616183342` (90-day retention).

## GPU allocation

The metadata requests `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, and the push uses `--accelerator NvidiaTeslaT4`. This matches the [Kaggle CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). Kaggle requires a phone-verified account for any GPU, and an unverified account silently runs on CPU. Once the account owner completes phone verification at kaggle.com/settings, the same workflow will train the ~24.4M-parameter GPU profile without code changes.
