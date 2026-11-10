# ArtemisLM S3 joint instruction-tuning checkpoint

- Parent: F2, global step 17,500
- Parent SHA-256: c4a791f9941ef4488c74a51bf2a46fce40b0764aa8bd81683c35385939efd584
- Stage: response-only instruction SFT, 1,500 optimizer steps
- Final global step: 19,000
- Architecture: unchanged; 6,451,200 parameters
- Optimizer: retained in the full checkpoint for future resume
- Inference: optimizer and RNG state omitted from the inference-only file
- Dataset provenance: s3-mix-manifest.json
- Held-out capability diagnostics: capability-eval.json
- Smoke-test generation: s3-sample.txt
- Run metadata and hashes: s3-run-summary.json and SHA256SUMS
