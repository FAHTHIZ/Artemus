# Artemis Core Philosophy

**Status:** Foundational project invariant

## 1. This is our own model

Artemis is an independently developed foundation model. Every official foundation checkpoint must have a documented lineage beginning with random initialization in this project.

Prohibited shortcuts:
- Importing another model's weights and relabelling or repackaging them as Artemis.
- Quietly using a pretrained model as the hidden base of a claimed from-scratch checkpoint.
- Presenting another model's output, benchmark result, or capability as an Artemis result.

Published papers, open research, public benchmarks, and appropriately licensed data/code may inform our engineering when their terms are respected and their use is recorded. Learning from scientific literature is not the same as copying learned model weights.

External models may be used as independent evaluators and comparison points, not as Artemis parameters or default training-label generators.

## 2. Provenance is mandatory

Every checkpoint records random initialisation and seed; source-code commit; architecture and tokenizer versions; dataset manifests, immutable source revisions, licences, and hashes; training configuration and compute environment; parent checkpoint; evaluation results, known limitations, and artifact hashes.

A checkpoint with missing or ambiguous lineage is not an official Artemis release.

## 3. Capability per parameter

Optimise useful capability per stored parameter, active parameter, training compute, inference FLOPs, memory, latency, and energy.

Always distinguish:
- **Total parameters:** all learned weights stored by the model.
- **Active parameters:** weights used for a token or forward pass.
- **Inference memory:** weights, KV cache, activations, and runtime overhead.
- **Compute per token:** measured operations and throughput, not theoretical FLOPs alone.

A sparse model with one billion total weights and two hundred million active weights is compute-sparse, not a two-hundred-million-parameter model. Reports must state both.

## 4. Ambitious frontier target, empirical claims

The long-term target is frontier-class performance in reasoning, coding, mathematics, multilingual communication, knowledge work, and reliable tool use, while keeping the deployed model materially smaller and cheaper than the largest dense systems.

This is an engineering ambition, not a promised outcome. We will not claim parity with GPT-6.1 Sol, GPT-6 Astra, Claude Fable 5.1, or another frontier model based on design intent or falling training loss. Parity can only be claimed dimension by dimension, under reproducible and comparable evaluations.

A compact specialist that is excellent at software engineering is a valid milestone. It must not be marketed as general frontier intelligence unless evidence supports that broader claim.

## 5. Intellectual openness and candour

Artemis should support broad, direct, intellectually honest discussion. It should avoid arbitrary viewpoint-based refusals, evasive boilerplate, and hiding uncertainty behind confident language.

"Uncensored" here means candid inquiry and minimal arbitrary topic suppression. It does not mean disabling privacy protections, lawful access controls, consent boundaries, or safeguards against facilitating serious harm. Restrictions must be explicit, proportionate, context-aware, and testable rather than used to avoid ordinary difficult discussion.

The model should distinguish discussing a subject from taking action; explaining a risk from enabling it; fictional or academic exploration from operational harmful instructions; and uncertainty from verified fact.

When assistance is limited, explain the concrete boundary and remain useful where safe alternatives exist.

## 6. Data integrity

Use only data with a documented permission or licence basis appropriate to the intended use. Publicly accessible is not the same as unrestricted.

Default policy:
- No private conversations without explicit consent and a defined use.
- No credentials, secrets, or unnecessary personal data.
- No scraping around access controls, paywalls, or repository restrictions.
- No unlicensed copying of books, code, or proprietary corpora.
- No external-model-generated training corpus by default. Synthetic tasks should be deterministic, human-authored, or verified against executable outcomes.
- Preserve attribution and source-level obligations.

When data rights are unclear, quarantine it instead of assuming permission.

## 7. Continuous improvement without uncontrolled self-modification

Conversation feedback may create candidate memories or training examples. It does not directly rewrite production weights.

Every weight update requires a provenance-tracked dataset change, reproducible training run, independent evaluation, regression and safety tests, and a retained prior checkpoint with a rollback route.

## 8. Evidence before promotion

A candidate is better only if it improves relevant task metrics without unacceptable regressions. Preserve failed runs and negative findings. Do not cherry-pick prompts, benchmark versions, or seeds.

Be ambitious in goals and unsentimental about results.
