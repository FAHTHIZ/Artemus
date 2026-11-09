# Artemis — Model Design and English-only flagship research

## 1. Identity and initialization

Artemus is the complete cognitive-system research project. ArtemisLM is its experimental decoder-only neural model. Every official foundation checkpoint starts from random initialization within this repository. External pretrained weights are not converted, relabelled or distilled into an official foundation checkpoint.

## 2. Implemented architecture options

The current research implementation includes efficient dense Transformer components: RMSNorm, causal scaled-dot-product attention, grouped-query attention, RoPE, optional query/key normalization, SwiGLU, factorised embeddings and an incremental KV cache. Each feature remains a hypothesis until quality, memory, throughput and latency are compared at matched budgets.

Do not add sparse experts, latent-attention variants or recurrent hybrids solely because they are fashionable. Establish stable dense baselines and run controlled ablations first.

## 3. Tokenizer and language scope

ArtemisLM training is English only. The tokenizer is fitted only on approved English training documents. Measure coverage for general prose, technical writing, code, mathematics, shell commands, identifiers, URLs and structured data. Keep validation/test text out of tokenizer fitting.

## 4. Capability priorities

Prioritise English software engineering, mathematics, logical reasoning, science/technical comprehension, general knowledge, instruction following, tool use, factuality and long-horizon reliability. OASIS capabilities are evaluated separately from intrinsic model-weight capability.

## 5. Training curriculum

1. **Foundation pretraining:** general English FineWeb plus educational English FineWeb-Edu, starting from random weights.
2. **Domain adaptation:** reviewed English code, technical and mathematical text with general-English replay.
3. **Instruction SFT:** OASST1 English conversations and Aya records restricted to `eng`.
4. **Task SFT:** approved English math and Python tasks with protected tests.
5. **Verified outcomes and preferences:** defer until independent validators and the required objectives exist.

## 6. Data rights and provenance

FineWeb and FineWeb-Edu declare ODC-By 1.0; Common Crawl terms, attribution, residual privacy and content risks still apply. For OSCR, preserve original per-file licences and repository provenance. Public availability does not automatically confer permission to train on or redistribute every source.

## 7. Scaling methodology

Build a Pareto curve across parameters, active parameters, training FLOPs, unique tokens, inference bytes, latency, energy, cost and held-out capability. Compare model sizes at matched token or FLOP budgets. A small model cannot be declared frontier-equivalent from parameter counts, loss curves or one benchmark.

## 8. Checkpoint integrity

Record source revisions, language-gate statistics, hashes, tokenizer, model config, optimizer state, accelerator, steps, evaluation inputs and outputs. Promote only after reproducible held-out improvement.
