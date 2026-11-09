# Artemis — English-only training strategy

## 1. Non-negotiable lineage

Official ArtemisLM foundation checkpoints begin from random initialization in this repository. External models may inform architecture research or produce clearly labelled teacher candidates under their terms, but pretrained weights must never be disguised as Artemis foundation weights.

## 2. Data pipeline

Every input must be pinned, licence-reviewed, deduplicated and provenance-preserving. Foundation text is strictly English: records need an explicit English language label and a confidence score of at least 0.70. Aya SFT accepts only `language_code=eng`. Missing or non-English labels are rejected before tokenizer fitting and before training.

Keep benchmark test splits protected. Separate web pretraining text, assistant instructions, mathematics, source code and preference pairs by objective. Do not flatten all data families into one undifferentiated text file.

## 3. Data families

- **General English text:** FineWeb and FineWeb-Edu are the active bounded pilot sources. Deduplicate them because they draw from overlapping web material.
- **English code:** use OSCR only with exact per-file and per-repository licence evidence, immutable commits and source hashes.
- **English mathematics:** GSM8K for verified task SFT; OpenWebMath can be explored separately as domain-adaptation text, not verified answers.
- **English conversations/instructions:** reviewed OASST1 and Aya filtered to `eng`.
- **Human preferences:** HH-RLHF is deferred until a preference/reward-model objective exists; do not mix preference pairs into ordinary response-only SFT.
- **Teacher outputs:** outputs from released models may be candidate examples only, with exact model/version/terms, independent verification, human review and provenance.

## 4. Training stages

### Stage A — pipeline correctness
Test tokenization, document boundaries, loss masking, validation isolation, checkpoint recovery, hashes and provenance.

### Stage B — English foundation pretraining
Start from random weights. Train on a deduplicated general-English / educational-English mixture. Measure source token shares using the Artemis tokenizer.

### Stage C — domain adaptation
Add licensed English code, technical material and math text under a bounded mixture that preserves general-English replay.

### Stage D — instruction SFT
Use human-reviewed English data only. Train response-only loss and preserve complete prompt groups across splits.

### Stage E — verified capabilities
Use exact-answer math and test-executed code. Candidate programs run in a resource-limited, network-disabled sandbox.

### Stage F — preference and outcome optimisation
Begin only after the base model is coherent and instruction following is stable. Require human preference data, independently checked rewards, rollback and anti-reward-hacking analysis.

### Stage G — evaluation and release
Compare matched checkpoints, publish regressions, preserve rollback points and require protected holdouts.

## 5. Mixture discipline

Initial FineWeb/FineWeb-Edu document quotas are 80/20. This is a pilot setting. Track actual unique tokens, repeated tokens, token fertility, per-source losses and downstream task outcomes. Do not combine DCLM, Dolma or SlimPajama with FineWeb until overlap and rights are checked. Choose one alternative at a time for matched-budget comparisons.

## 6. Tokenizer and English coverage

Train SentencePiece on training text only. Measure English general prose, technical terms, source-code identifiers, mathematical notation, URLs, shell commands and structured text. Keep validation/test documents out of tokenizer fitting. Version every tokenizer with its exact corpus manifest.

## 7. Evaluation

Use protected English language, instruction, reasoning, mathematics, coding, tool-use and long-horizon tasks. Report base model and integrated OASIS separately. Keep public benchmark scores separate from project-authored private holdouts; record compute, inference budget, prompts, evaluator version and failures.
