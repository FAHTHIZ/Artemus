# ArtemisLM F3 continued-pretraining checkpoint

- Parent: F2, global step 17,500
- Parent SHA-256: c4a791f9941ef4488c74a51bf2a46fce40b0764aa8bd81683c35385939efd584
- Stage: continued next-token pretraining, 4,000 optimizer steps
- Final global step: 21,500
- Architecture: unchanged; 6,451,200 parameters
- Mixture: 85% fresh FineWeb-Edu shard 008; 15% FineMath-4+ shard 00012 replay
- Optimizer: retained in the full checkpoint for future resume
- Inference: optimizer and RNG state omitted from the inference-only file
- Dataset/source provenance: f3-mix-manifest.json and f3-source-shards.json
- Held-out evaluation: lm-eval.json
- Run metadata and hashes: f3-run-summary.json and SHA256SUMS
