ArtemisLM F2 continued-pretraining checkpoint (Experiment 0016), continued from F1 (`f1-math-6m-step12500`).

- 6,451,200 parameters; same architecture as F0/F1
- +5,000 steps (12,500 → 17,500; batch 16 × 256 ≈ 20.5M tokens) on a 207M-token mix: FineWeb-Edu sample-10BT shards 000 + 007 (edu score ≥ 3.5, deduplicated against shard-000 train and all validation sets) + FineMath-4+ replay
- LR: 200-step warmup to 8e-4 and cosine (planned 10k steps); from step 16,250, an early cosine anneal from 1.94e-4 to 8e-5 at 17,500
- Mixed validation loss 3.796 (13,500) → 3.734 (16,250) → **3.664** (17,500)
- Held-out bits/byte: **FineWeb-Edu 1.321**, FineMath 1.507 (F1 1.350 / 1.508; F0 1.346 / 1.887). Best base model so far
- Base language model (no chat tuning); see `f2-samples.txt`
