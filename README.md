# Artemus checkpoints

This orphan branch stores trained ArtemisLM weights so `main` stays free of large binaries.
Each folder is one promoted training stage, with its held-out evaluation, samples, data manifest and SHA-256 sums.

| Folder | Stage | Params | Held-out bits/byte (FineWeb-Edu) | Held-out bits/byte (FineMath) |
|---|---|---:|---:|---:|
| `f0-edu-6m-step10000/` | F0 foundation, FineWeb-Edu score >= 3.5 (Experiment 0011) | 6,451,200 | 1.346 | 1.887 |
| `f1-math-6m-step12500/` | F1 math adaptation, 40% FineMath-4+ / 60% edu replay (Experiment 0012) | 6,451,200 | 1.350 | 1.508 |
| `s1-aya-6m-step13300/` | S1 response-only English instruction SFT on Aya (Experiment 0013) | 6,451,200 | 1.479 | 1.692 |
| `s2-task-6m-step14800/` | S2 response-only task SFT on OASST1 + GSM8K train + MBPP train (Experiment 0014) | 6,451,200 | 1.410 | 1.632 |
| `f2-edu2-6m-step17500/` | F2 continued pre-training from F1 on FineWeb-Edu shards 000+007 + FineMath (Experiment 0016) | 6,451,200 | 1.321 | 1.507 |
| s3-joint-6m-step19000-run-42733213811/ | S3 response-only joint English instruction SFT | 6,451,200 | — | — |

`*-inference.pt` drops optimizer state (for `artemus-generate` / `artemus-lm-eval`); the full `.pt` resumes with `artemus-train --resume`.

```bash
git clone --branch checkpoints --single-branch https://github.com/FAHTHIZ/Artemus.git artemus-checkpoints
```
