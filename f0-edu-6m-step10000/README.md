ArtemisLM F0 foundation checkpoint, trained from random initialization (Experiment 0011).

- 6,451,200 parameters: 6 layers, width 256, GQA 4/2 heads, RoPE, QK-norm, SwiGLU; 256-token context
- 10,000 steps (~41M tokens) on 88,355 English FineWeb-Edu documents with educational score >= 3.5 (ODC-By 1.0, Common Crawl terms apply)
- Validation loss 3.749; held-out 1.346 bits/byte (FineWeb-Edu), 1.887 bits/byte (FineMath, unseen)
- Base model only: fluent-looking but repetitive and not factual. Not an assistant.

Files: `artemus-f0-edu-6m-step10000.pt` (checkpoint incl. optimizer state and embedded tokenizer; resumable with `artemus-train --resume`), `tokenizer.model`, held-out eval JSON, samples, data manifest, SHA256SUMS.

Data attribution: FineWeb-Edu by Hugging Face (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), ODC-By 1.0.
