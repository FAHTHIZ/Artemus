# Artemis — Model Design and Research Plan

**Status:** Experimental research program; final dimensions are not committed.  
**Invariant:** Foundation weights begin from random initialization in the Artemis project.

## 1. Independent training

Artemis is our own independently trained model. No foreign pretrained checkpoint, hidden pretrained component, or renamed model weights may be presented as Artemis. External systems can be research references and independent evaluators, not hidden bases or default sources of training answers.

See [Core Philosophy](../constitution/core-philosophy.md).

## 2. Capability objective

Target frontier-class capability per parameter and per unit of compute, with software engineering as the first specialist strength and broad reasoning as the long-term objective. This is a hypothesis to test, not a guaranteed outcome.

Always report total parameters, active parameters, training/inference compute, memory, tokens by source, and capability-specific scores.

## 3. Baseline and research candidates

Start with the working decoder-only dense Transformer for pipeline validation. Research a more efficient dense baseline before adding routing complexity.

Candidate components for controlled ablations:
- RMSNorm, RoPE, SwiGLU, and stable initialisation.
- Grouped-query attention to reduce KV-cache costs.
- Fused attention where supported.
- Weight tying and context packing.
- Fine-grained sparse Mixture-of-Experts with measured router balance.
- Weight-shared/recurrent depth or adaptive inference compute.
- Retrieval/OASIS as a separate system-level capability layer.

Do not adopt an architecture because it is fashionable. Use matched compute and quality evaluations.

## 4. Tokenizer

The byte tokenizer is a pipeline fallback, not the production design. SentencePiece is the current experiment. Evaluate token fertility and round-trip behaviour across English, Bengali, Banglish, code, maths, and shell/structured text. Dedicated control tokens must remain atomic at all prompt positions.

## 5. Training lifecycle

1. Data audit and tokenizer evaluation.
2. Random-init foundation pretraining on licensed, deduplicated general, technical, code, mathematics, and multilingual data.
3. Supervised instruction tuning using reviewed examples and response-only masks.
4. Verified coding, math, and tool trajectories with exact validators.
5. Outcome-based optimisation where rewards can be validated.
6. High-quality anneal, regression evaluation, and release.

OASST1 is a first conversation experiment, not the broad foundation corpus.

## 6. Compact-model research

Compare dense and sparse options at matched active compute. A sparse model can store more total parameters than it activates for each token; that may improve compute efficiency but does not reduce storage to the active count.

See [Compact Frontier-Capability Strategy](compact-frontier-strategy.md).

## 7. Evaluation

Evaluate raw Artemis and Artemis + OASIS separately. Measure general language, coding, repository tasks, mathematics, multilingual competence, factuality, tool use, long-horizon reliability, and inference efficiency.

Never claim parity or superiority without same-version, same-harness evidence. See [Frontier Comparison and Evaluation Plan](../evaluation/frontier-targets.md).
