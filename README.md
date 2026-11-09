# Artemus checkpoints

This orphan branch stores trained ArtemisLM weights so `main` stays free of large binaries.
Each folder is one promoted training stage, with its held-out evaluation, samples, data manifest and SHA-256 sums.

| Folder | Stage | Params | Held-out bits/byte (FineWeb-Edu) |
|---|---|---:|---:|
| `f0-edu-6m-step10000/` | F0 foundation, FineWeb-Edu score >= 3.5 (Experiment 0011) | 6,451,200 | 1.346 |

`*-inference.pt` drops optimizer state (for `artemus-generate` / `artemus-lm-eval`); the full `.pt` resumes with `artemus-train --resume`.

```bash
git clone --branch checkpoints --single-branch https://github.com/FAHTHIZ/Artemus.git artemus-checkpoints
```
