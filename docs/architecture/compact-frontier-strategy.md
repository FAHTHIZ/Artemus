# Compact Frontier Flagship — Research and Engineering Strategy

**Research snapshot:** 2026-11-02  
**Project invariant:** Every official Artemis foundation checkpoint begins from random initialization within this repository.  
**Objective:** Maximise general capability per total parameter, active parameter, training FLOP, inference byte, latency, energy, and cost.

## 1. What “frontier flagship” means here

The ambition is not merely to build a small model that performs well on toy benchmarks. Artemis must eventually demonstrate strong and robust software engineering, mathematical and general reasoning, multilingual competence, knowledge work, tool use, and long-horizon task execution.

The project will measure the base model and the complete OASIS system separately. A tool-augmented coding success must not be credited as raw-model ability.

We will not lock a final parameter count before scaling experiments. Build a Pareto curve across total/active parameters, quality, memory, throughput, energy, and cost. A compact model that loses important general capabilities is not the flagship just because it is small.

## 2. Current baseline and honest interpretation

The current 1.085M-parameter OASST1 response-only model produces incoherent text. The next CPU run mixes up to 30,000 OASST1 conversations with GSM8K train-split arithmetic/reasoning and MBPP official training-ID Python tasks, using a 1.25M-scale random-init model for 8,000 steps.

That next run is an **initial multi-task supervised experiment**, not broad foundation pretraining. The training sequence sees only a tiny number of sampled token positions compared with the tens of billions of tokens used in modern foundation-model research. It can test the pipeline, data plumbing, loss masking, and checkpoint lineage; it cannot certify a flagship.

## 3. Research findings that influence our design

### 3.1 Architecture and scaling must be co-optimised

The ICLR 2026 paper *Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs* trains more than 200 models from 80M to 3B parameters on 8B–100B tokens. It explicitly studies hidden size, MLP-to-attention allocation, and GQA as variables that affect both accuracy and inference cost. Its authors report architecture-aware choices improving accuracy and throughput under matched budgets.

Implication: do not choose width, depth, attention heads, or MLP ratio by folklore. Run controlled architecture sweeps at multiple model sizes and fit an empirical scaling model for our compute and target workload.

Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html

### 3.2 Data quality changes effective model size

The ICLR 2026 paper *Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining* introduces a data-quality term into scaling-law analysis. Its central engineering message is that noise, redundancy, and corruption change the effective value of a corpus; raw token count is not the same as useful training information.

Implication: clean, deduplicate, score, and source-track data before scaling tokens. Measure quality by domain, language, licence, and source. Preserve document-level provenance and held-out evaluation.

Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html

### 3.3 More epochs are not equivalent to fresh data

The NeurIPS 2026 study *Bridging Compute- and Data-Optimal Pretraining* models the diminishing effectiveness of repeated/derived tokens as corpus supply becomes constrained. It reports that token effectiveness depends on model size, tokens-per-parameter, and the amount of derived data.

Implication: record unique tokens, repeat count, and data expansion separately. Repeating a tiny corpus can validate code and lower training loss; it does not reproduce a large fresh corpus.

Paper: https://www.microsoft.com/en-us/research/publication/bridging-compute-and-data-optimal-pretraining/

### 3.4 Multilingual data needs its own scaling strategy

Google's ATLAS work (ICLR 2026) reports 774 training runs across models from 10M to 8B parameters, 400+ languages, and 48-language evaluations. It estimates transfer between language pairs and shows that language mixture selection is an empirical scaling problem.

Implication: English-only token ratios cannot determine the best Bangla/English mixture. Track language IDs, token fertility, and held-out performance per language. Bangla and Banglish need dedicated evaluation and sufficient high-quality conversational, educational, and technical content.

Research overview: https://research.google/blog/atlas-practical-scaling-laws-for-multilingual-models/

### 3.5 Small-model reasoning is possible, but robustness remains a challenge

ThinkSLM evaluated 72 small models across 17 reasoning benchmarks and reports that training methods and data quality matter substantially, while larger models generally retain better robustness to adversarial perturbations and intermediate reasoning.

Implication: include adversarial paraphrases, multi-step tasks, and out-of-distribution tests; do not treat small-model benchmark wins as proof of robust reasoning.

Paper: https://aclanthology.org/2025.emnlp-main.1659/

## 4. Architecture research ladder

Keep the existing dense baseline as a reference. Every change must be ablated independently before combinations are selected.

### A. Efficient dense Transformer (active implementation)

- RMSNorm, causal scaled-dot-product attention, and SwiGLU.
- RoPE as an alternative to learned absolute position embeddings.
- Grouped-query attention (GQA) with query heads sharing fewer key/value heads.
- Optional query/key RMSNorm as a stability hypothesis.
- Factorised token embeddings/output projection to trade embedding-table parameters for input/output projection compute.
- Weight tying, robust initialization, gradient clipping, and carefully tuned schedules.

The present V2 implementation is a research candidate. It passes forward/backward and checkpoint-compatibility tests and now includes incremental KV caching. Persistent cached keys/values use `n_kv_head` heads; the current SDPA compatibility path temporarily expands them for attention. Cached prefill and continuation support RoPE offsets, learned absolute positions, multi-token chunks, and context-window rollover. A tiny random-init CPU benchmark showed a 1.33× cached/uncached ratio, but that is a systems smoke result—not a trained-model or accelerator guarantee. Cache concatenation and temporary GQA expansion remain optimization targets.

### B. Attention memory alternatives

Research Multi-head Latent Attention (MLA) as a later candidate. It compresses key/value state into a latent representation rather than retaining conventional per-head K/V tensors. Compare quality, decode bandwidth, cache bytes, implementation complexity, and compatibility with available kernels. Do not copy a proprietary implementation; implement from public papers and test from scratch.

### C. Sparse expert models

MoE separates total stored weights from per-token active weights. It may improve capability per unit of arithmetic, but all experts must be available somewhere and routing/communication can dominate at small scale.

Do not implement MoE on the current tiny CPU model. First establish a meaningful dense baseline and accelerator-backed scaling runs. Then compare:
- Fine-grained experts and shared dense layers.
- Top-k routing, router balance, expert capacity, and overflow.
- Expert parallelism and storage/communication cost.
- Total versus active parameters, actual tokens/s, memory, and task accuracy.

Recent ICML 2026 work on Mixture of Slimmable Experts (MoSE) is a future candidate: it combines sparse expert selection with variable expert width, allowing a continuum of compute/accuracy choices. It is a research direction, not an adopted architecture.

Paper: https://proceedings.mlr.press/v306/tastan26a.html

### D. Adaptive depth and inference-time reasoning

Investigate recurrent/weight-shared blocks, early exit, and adaptive compute after dense baselines are stable. Extra inference compute can improve difficult tasks, but it increases latency and must be reported as a budget, not described as free intelligence.

### E. Factorised embeddings

The token/output vocabulary matrix can dominate small models. With vocabulary (V), model width (d), and factorised embedding width (d_e), the table changes from (Vd) to (Vd_e), with roughly (2dd_e) extra projection parameters.

At (V=4096,d=128,d_e=64), the table saves 262,144 parameters and the two projections add 16,384, for a net reduction of 245,760 parameters. This does not guarantee quality retention; the embedding bottleneck may hurt rare code identifiers, mathematics, and multilingual token fragments.

### F. Retrieval and agent tools are a separate capability layer

Use OASIS for approved-source retrieval, repository indexing, test execution, compiler feedback, task planning, checkpoints, and verification. Report base-model and agent-system results separately. External memory can make a compact system more useful without pretending it enlarged the model's intrinsic parametric knowledge.


### G. Hybrid linear/full attention (future research only)

Gated DeltaNet, Native Hybrid Attention, and dynamic full/sliding-window attention offer possible long-context efficiency improvements. They are not automatic wins: linear/recurrent state can lose precise retrieval or multi-step composition that full attention preserves. Any candidate must be trained from Artemis random initialization and compared on long-context needle retrieval, multi-document synthesis, exact recall, reasoning, and generation quality at matched compute.

**Lineage constraint:** do not convert or distil an external pretrained model into the claimed Artemis foundation checkpoint. Published architectural ideas may be independently implemented, but the official foundation weights must remain ours from random initialization. If an architecture only becomes viable through transferring another provider's learned weights, it is not eligible as the Artemis foundation model.

## 5. Training curriculum — from random weights to a flagship

### Stage 0 — data rights, provenance, and tokenizer
Create source manifests before ingestion. Preserve immutable revisions, licence/attribution, language, document IDs, hashes, opt-out state, and preprocessing decisions. Train tokenizers on training data only. Evaluate English, Bangla, Banglish, code, maths, shell, and structured text for fertility and round-trip correctness.

### Stage 1 — broad foundation pretraining
Train next-token prediction from random initialization on a broad, balanced, deduplicated and legally reviewed corpus:
- High-quality general prose and reference material.
- Technical/scientific and educational sources.
- Properly licensed source code with per-repository/file provenance.
- Bengali and other target-language corpora, sampled deliberately.
- Maths, structured reasoning, and selected scientific text.

FineWeb is a large candidate for general English pretraining, but its ODC-By 1.0 attribution conditions and Common Crawl terms apply. The dataset card acknowledges residual PII and undesirable content; run additional privacy and quality filtering. FineWeb is mostly web text, not a sufficient source-code corpus. See the data registry.

### Stage 2 — domain-adaptive continued pretraining
Continue next-token training on curated code, technical documentation, mathematics, and target languages. Keep the general corpus in the mixture to reduce catastrophic forgetting. Track per-domain and per-language validation losses.

### Stage 3 — supervised instruction tuning
Train on reviewed prompts and high-quality human-authored responses. Use response-only loss masks where appropriate. Keep full conversation trees together across splits. Include multi-turn tool schemas and reliable format following.

### Stage 4 — verified capability training
Train on tasks with machine-checkable outcomes: compilers, unit tests, theorem/exact-answer checkers, numerical validators, and structured API validators. Use sandboxed execution. Public benchmark test examples must remain excluded. Synthetic tasks should be deterministic, human-authored, or independently verified—not copied from frontier-model answers.

### Stage 5 — outcome optimisation
Where reward is verifiable, optimise for task outcomes: tests pass, builds succeed, exact results match, and tool calls satisfy schemas. Add reward-model or preference methods only with carefully audited labels, held-out evaluators, and reward-hacking tests.

### Stage 6 — high-quality anneal and capability balancing
Use a final high-quality mixture to balance general language, code, reasoning, mathematics, and multilingual ability. Prevent coding gains from silently degrading conversation, factuality, or Bangla. Evaluate catastrophic forgetting after every phase.

## 6. Scaling experiments before choosing the flagship size

Run matched compute experiments in a staged ladder, for example 10M, 30M, 100M, 300M, and larger only when accelerator budget allows. For each size, compare:
- Multiple widths/depths and MLP-to-attention ratios.
- Token budget and unique-data budget.
- Data quality/filtering levels.
- GQA ratios and factorised embedding dimensions.
- Dense model versus sparse variants when the model is large enough to justify routing.

Fit empirical curves for validation loss and task score against compute, tokens, total parameters, active parameters, and inference cost. The eventual flagship size should be selected from the measured Pareto frontier. A small CPU runner cannot run these serious scale sweeps; it can validate the pipeline only.

## 7. Compute reality and infrastructure plan

GitHub-hosted CPU runners are useful for unit tests, dataset preparation, tokenizer experiments, parameter profiling, and tiny smoke runs. They are not a path to tens of billions of unique pretraining tokens or frontier-class training.

A later GPU stage needs:
- A reproducible accelerator environment and tested mixed precision.
- Efficient batched data loading/packing, distributed training where appropriate, and checkpoint/resume.
- Optimizer-state/memory planning, monitoring, and fault recovery.
- Costed token/FLOP budgets and a data pipeline sized to avoid idle accelerators.
- Model, optimizer, scheduler, RNG, tokenizer, and data-position checkpointing.

The user's phone can control development and review. It is not the training accelerator. Any future cloud/GPU use must be explicit and costed; no free quota is assumed.

## 8. Promotion milestones

1. **Reproducible pipeline:** tests, source checksums, random-init provenance, and resumable checkpoints.
2. **Coherent small model:** held-out prompts produce readable, relevant answers.
3. **Specialist competence:** unseen code/math tasks pass verifiable tests at a measured rate.
4. **Useful multilingual baseline:** Bangla/Banglish are tested and not merely tokenized.
5. **Scaling evidence:** size/data/architecture sweeps produce reproducible scaling curves.
6. **Efficient candidate:** dense/sparse and cache alternatives improve the measured quality-cost frontier.
7. **Long-horizon agent:** OASIS performs unseen repository tasks with reliable verification.
8. **Flagship comparison:** independent comparable evaluations across the complete scorecard.

No design paper, falling loss curve, toy demo, or single public benchmark is enough to label Artemis frontier-capable.
