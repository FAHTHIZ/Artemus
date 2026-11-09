ArtemisLM F1 math domain-adapted checkpoint (Experiment 0012), continued from F0 (`f0-edu-6m-step10000`).

- 6,451,200 parameters: 6 layers, width 256, GQA 4/2 heads, RoPE, QK-norm, SwiGLU; 256-token context
- +2,500 steps (~10M tokens) on a 40% FineMath-4+ / 60% FineWeb-Edu (score >= 3.5) replay mix, LR 6e-4 -> 6e-5 cosine
- Held-out bits/byte: FineMath 1.887 -> **1.508** (-20%), FineWeb-Edu 1.346 -> 1.350 (no meaningful forgetting)
- Base model only. It imitates maths-text formatting but cannot do arithmetic. Not an assistant.

Files: full resumable `.pt`, `-inference.pt` (optimizer/RNG stripped), `tokenizer.model`, eval JSON, samples, data manifests, SHA256SUMS.

Data attribution: FineMath by Hugging Face (https://huggingface.co/datasets/HuggingFaceTB/finemath, ODC-By 1.0); FineWeb-Edu by Hugging Face (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, ODC-By 1.0). Common Crawl terms apply.
