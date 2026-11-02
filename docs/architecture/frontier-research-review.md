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
| ICLR 2026, *Pretraining Scaling Laws for Generative Evaluations* | Fits scaling laws to pass@k on math/software-engineering tasks and studies how inference budget changes observed capability. | Track held-out generative task outcomes and pass@k across model size, training compute, and inference budget; loss alone is not a sufficient proxy. |
| ICLR 2025, *Inference Scaling Laws* | Finds smaller models with additional search/test-time compute can occupy better cost/performance trade-offs on selected reasoning tasks. | Explore verifier-guided sampling/search only after the base model is coherent; report all extra tokens and compute. |
| ICLR 2025, *Language Models Scale Reliably with Overtraining and on Downstream Tasks* | Studies scaling beyond compute-optimal token/parameter ratios and relates pretraining loss to downstream task performance. | Fit both compute-optimal and inference-efficient/overtrained regimes; measure downstream outcomes directly rather than extrapolating from loss alone. |
| ICLR 2026, *How Text Quality Interventions Reshape Neural Scaling Laws* | Large-scale experiments show filtering, deduplication, and text-quality interventions can change fitted scaling-law coefficients and exponents. | Treat the data pipeline as an experimental variable; preserve raw/filtered comparisons and quality labels. |
| ICLR 2025, *Gated Delta Networks* | Gating and delta-rule updates can improve linear-attention memory control and retrieval; hybrid variants combine recurrent/linear layers with sliding-window attention. | Keep as a long-context research branch after dense baselines; test retrieval and reasoning, not only throughput. |
| ACL 2026, *Native Hybrid Attention* | Combines recurrent long-term state with a local sliding-window token cache inside a unified attention operation. | Consider a from-scratch hybrid candidate only if long-context tests show a useful quality/latency trade-off. |
| EMNLP 2026, *Switch Attention* | Dynamically routes between full and sliding-window attention at token/layer granularity. | Treat dynamic routing as a later hypothesis; measure router overhead and global-recall regressions. |
| OpenAI SWE-bench Verified analysis (2026) | Benchmark contamination weakens the value of Verified as a frontier coding measure; SWE-bench Pro is recommended. | Maintain private holdouts and use current, contamination-aware repository evaluations. |

## Source links

- Architecture-aware scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html
- Data-quality scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html
- Compute-data scaling: https://www.microsoft.com/en-us/research/publication/bridging-compute-and-data-optimal-pretraining/
- Multilingual ATLAS: https://research.google/blog/atlas-practical-scaling-laws-for-multilingual-models/
- ThinkSLM: https://aclanthology.org/2025.emnlp-main.1659/
- MoSE: https://proceedings.mlr.press/v306/tastan26a.html
- Generative evaluation scaling: https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d7fbeddfd9bb6dace146cc4fa02ff0-Abstract-Conference.html
- Inference-time scaling: https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html
- Overtraining and downstream scaling: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a91869936a63d814971b6423990ecf6e-Abstract-Conference.html
- Text-quality interventions: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html
- Gated Delta Networks: https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html
- Native Hybrid Attention: https://aclanthology.org/2026.acl-long.176/
- Switch Attention: https://aclanthology.org/2026.emnlp-main.2079/
- SWE-bench Verified contamination: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/

## Interpretation

The literature does not yield a magic architecture that turns a small random-init model into a frontier flagship. The consistent direction is to co-optimise data quality, model size, architecture, training compute, and inference budget—and validate each choice on held-out tasks. Artemis will use those methods as research inputs while keeping its weights and training lineage independent.
