# Frontier Research Review — 2026-11-02

This document records research that materially informs the Artemis flagship strategy. It records findings and engineering implications; it does not assert that Artemis has reproduced the published results.

| Research | Finding relevant to Artemis | Engineering implication |
|---|---|---|
| ICLR 2026, *Scaling Laws Meet Model Architecture* | More than 200 model runs from 80M–3B parameters and 8B–100B tokens jointly study width, MLP/attention allocation, and GQA. | Architecture should be selected using matched-budget scaling experiments, not intuition. |
| ICLR 2026, *Scaling Laws Revisited: Modeling the Role of Data Quality* | Data quality changes the effective value of a token and belongs in scaling analysis. | Measure deduplication, corruption, source quality, and effective unique tokens. |
| NeurIPS 2026, *Bridging Compute- and Data-Optimal Pretraining* | Repeated/derived tokens have diminishing and model-dependent effectiveness. | Repeating a small dataset is a pipeline method, not a replacement for fresh data. |
| Google ATLAS (ICLR 2026) | 774 runs across 10M–8B models, 400+ languages, and 48 language evaluations model transfer and language mixture. | Bangla/English sampling needs language-specific measurement and controlled mixture experiments. |
| ThinkSLM (EMNLP 2025) | 72 small models across 17 reasoning benchmarks; data and training choices matter, and robustness remains a challenge. | Use verified reasoning curricula and adversarial/generalisation tests. |
| MoSE (ICML 2026) | Slimmable experts allow conditional compute through variable expert width as well as routing. | Consider only after a meaningful dense baseline and GPU-scale MoE comparison. |
| OpenAI SWE-bench Verified analysis (2026) | Benchmark contamination weakens the value of Verified as a frontier coding measure; SWE-bench Pro is recommended. | Maintain private holdouts and use current, contamination-aware repository evaluations. |

## Source links

- Architecture-aware scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html
- Data-quality scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html
- Compute-data scaling: https://www.microsoft.com/en-us/research/publication/bridging-compute-and-data-optimal-pretraining/
- Multilingual ATLAS: https://research.google/blog/atlas-practical-scaling-laws-for-multilingual-models/
- ThinkSLM: https://aclanthology.org/2025.emnlp-main.1659/
- MoSE: https://proceedings.mlr.press/v306/tastan26a.html
- SWE-bench Verified contamination: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/

## Interpretation

The literature does not yield a magic architecture that turns a small random-init model into a frontier flagship. The consistent direction is to co-optimise data quality, model size, architecture, training compute, and inference budget—and validate each choice on held-out tasks. Artemis will use those methods as research inputs while keeping its weights and training lineage independent.
