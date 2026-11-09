# Compact Frontier Flagship — Research and Engineering Strategy

**Research scope:** English-only model training  
**Project invariant:** Every official Artemis foundation checkpoint begins from random initialization within this repository.  
**Objective:** maximise capability per parameter, training FLOP, inference byte, latency, energy and cost.

## 1. What a flagship means

Artemis must eventually demonstrate strong software engineering, mathematics, general reasoning, English knowledge work, tool use and long-horizon reliability. The base model and complete OASIS system are measured separately. Tool-augmented success is not raw-model capability.

Do not lock a final parameter count before scaling experiments. Build a Pareto curve across total/active parameters, quality, memory, throughput, energy and cost. A compact model that loses important capability is not the flagship merely because it is small.

## 2. Honest baseline

The early 1M-parameter random-init models and the later task-SFT checkpoint are pipeline experiments, not useful foundation models. The recorded math and Python generations were incoherent. They validate data plumbing, loss masking, checkpoint lineage and recovery; they do not certify a capable assistant.

## 3. Research findings

### 3.1 Architecture-aware scaling

The ICLR 2026 paper *Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs* studies width, MLP/attention allocation and GQA across more than 200 models. Choose width, depth, attention heads and MLP ratio using matched-budget sweeps rather than folklore.

Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html

### 3.2 Data quality

The ICLR 2026 paper *Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining* makes data quality part of scaling analysis. Clean, deduplicate, score and source-track data before scaling. Measure quality by domain, licence and source.

Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html

### 3.3 Fresh data versus repetition

The NeurIPS 2026 study *Bridging Compute- and Data-Optimal Pretraining* models diminishing returns from repeated/derived tokens. Record unique tokens separately from repeated tokens. Repeating a small corpus is a pipeline method, not a substitute for fresh data.

Paper: https://www.microsoft.com/en-us/research/publication/bridging-compute-and-data-optimal-pretraining/

### 3.4 Small-model reasoning

ThinkSLM evaluates small models across reasoning tasks and finds that data and training choices matter while robustness remains difficult. Include adversarial paraphrases, multi-step tasks and out-of-distribution English tests.

Paper: https://aclanthology.org/2025.emnlp-main.1659/

## 4. Architecture research ladder

Keep the dense baseline as reference; ablate each change independently.

### A. Efficient dense Transformer
- RMSNorm, causal scaled-dot-product attention and SwiGLU.
- RoPE as an alternative to learned absolute positions.
- Grouped-query attention and optional query/key RMSNorm.
- Factorised embeddings, weight tying, robust initialisation and tuned schedules.
- Measure quality, cache bytes, throughput and peak memory rather than assuming a theoretical win.

### B. Attention-memory alternatives
Research Multi-head Latent Attention from public papers; compare quality, decode bandwidth, cache bytes and kernel compatibility. Do not copy proprietary implementations.

### C. Sparse experts
MoE may improve arithmetic efficiency but stores all experts and introduces routing/communication costs. Do not implement MoE on the current tiny CPU baseline. Establish an accelerator-backed dense baseline first.

### D. Adaptive inference
Consider recurrent blocks, early exit and verifier-guided test-time search only after base coherence. Report extra latency and compute explicitly.

### E. Factorised embeddings
At vocabulary V, model width d and embedding width d_e, factorisation reduces the vocabulary table from Vd to Vd_e but adds projections. It can constrain rare technical terms and code identifiers. Retain it only if matched-quality evaluation supports it.

### F. Retrieval and agent tools
Use OASIS for approved-source retrieval, repository indexing, tests, compiler feedback, planning, checkpoints and verification. Report base-model and agent-system results separately.

### G. Hybrid attention
Gated DeltaNet, Native Hybrid Attention and dynamic local/full attention are later hypotheses. Test long-context retrieval, synthesis and reasoning at matched compute; do not infer gains from throughput alone.

## 5. English training curriculum

### Stage 0 — data rights and tokenizer
Preserve immutable revisions, licences, attribution, document IDs, hashes, opt-out state and preprocessing decisions. Train tokenizers only on approved English training data.

### Stage 1 — foundation pretraining
Start from random weights on deduplicated general English and educational English. The active pilot uses FineWeb and FineWeb-Edu with explicit English-language and confidence gates. Both declare ODC-By 1.0; Common Crawl terms, attribution, residual privacy and content risks require review.

### Stage 2 — domain adaptation
Add reviewed English code, technical documentation and mathematical text. Preserve a general-English replay fraction and track per-domain validation losses.

### Stage 3 — instruction SFT
Use reviewed English prompts and human-authored responses. Aya is restricted to `eng`; OASST1 is reviewed English. Keep prompt/conversation groups together across splits.

### Stage 4 — verified capabilities
Train against machine-checkable outcomes: exact math, compilers, unit tests and structured APIs. Use isolated sandboxes and protected benchmark tests.

### Stage 5 — preference and outcome optimisation
Use preference data only with a dedicated objective and audited labels. Require held-out evaluators and reward-hacking checks.

### Stage 6 — high-quality anneal
Balance English general language, code, reasoning, mathematics and tool-use performance. Evaluate forgetting after every phase.

## 6. Scaling experiments

Run matched-compute experiments in a staged size ladder. Compare widths/depths, token budgets, data-quality levels, GQA ratios and embedding dimensions. Fit empirical curves for validation loss, task scores, training FLOPs and inference cost. CPU runners validate mechanics; they do not deliver frontier pretraining scale.

## 7. Compute and promotion

GitHub-hosted CPUs support tests, data preparation and tiny smoke runs. Serious scaling requires allocated accelerators, efficient input pipelines, checkpoint/resume, memory planning and costed token/FLOP budgets.

Promotion requires reproducible pipeline, coherent English generation, specialist competence on unseen tasks, scaling evidence, efficient inference and reliable long-horizon agent performance. No paper, loss curve, toy demo or single public benchmark is sufficient.
