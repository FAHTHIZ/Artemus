# Frontier Research Review — English model systems

This document records research that informs Artemis engineering; it does not assert that Artemis has reproduced published results.

| Research area | Finding relevant to Artemis | Engineering implication |
|---|---|---|
| Architecture-aware scaling | Width, depth, attention and MLP allocation interact with compute and inference costs. | Run matched-budget sweeps instead of selecting architecture by intuition. |
| Data-quality scaling | Noise, redundancy and corruption change the useful value of training tokens. | Measure deduplication, source quality and effective unique tokens. |
| Compute/data trade-offs | Repeated and derived tokens have diminishing effectiveness. | Repetition validates mechanics; it is not corpus-scale training. |
| Small-model reasoning | Data and training choices matter; robustness remains difficult. | Use independently verified curricula and adversarial English holdouts. |
| Sparse experts | Routing may increase active capacity but adds storage and communication costs. | Consider only after an accelerator-backed dense baseline. |
| Generative evaluation scaling | Inference budgets change observed pass@k and capability. | Record search tokens and inference compute alongside outcomes. |
| Text-quality interventions | Filtering and deduplication change downstream utility. | Treat the data pipeline as an experimental variable. |
| Hybrid attention | Recurrent/local approaches may improve efficiency but regress recall. | Test long-context retrieval and composition at matched budgets. |
| Optimizers | Alternative optimizers may help in some regimes but require matched tuning. | Keep AdamW as baseline; compare alternatives independently. |

## Interpretation

There is no shortcut from a small random-init model to a frontier flagship. Co-optimise data quality, model size, architecture, training compute and inference budget, then validate decisions on held-out English tasks.

## Research links

- Architecture-aware scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html
- Data-quality scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html
- Compute/data scaling: https://www.microsoft.com/en-us/research/publication/bridging-compute-and-data-optimal-pretraining/
- ThinkSLM: https://aclanthology.org/2025.emnlp-main.1659/
- MoSE: https://proceedings.mlr.press/v306/tastan26a.html
- Gated Delta Networks: https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html
- Muon compute/time study: https://arxiv.org/abs/2505.02222
- Optimizer comparison: https://arxiv.org/abs/2509.02046
