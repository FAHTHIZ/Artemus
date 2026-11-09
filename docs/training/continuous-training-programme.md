# Artemus English-only continuous training programme

**Status:** Active training plan  
**Language scope:** English only  
**Purpose:** build a reproducible, source-audited training pipeline and measure general capability per compute budget.

## 1. Current checkpoint

The previously documented compact checkpoints are early random-initialised task-SFT experiments on OASST1, GSM8K and MBPP. They are useful for testing the trainer and inference interfaces, but their recorded mathematics and Python generations remain incoherent. They are not broad foundation-pretrained models and must not be represented as useful general assistants.

The next official foundation run starts from fresh random initialization with a tokenizer trained only on the approved English foundation mixture.

## 2. Training stages

| Stage | Method | English data | Promotion gate |
|---|---|---|---|
| F0 | Foundation next-token pretraining | FineWeb + FineWeb-Edu | Held-out loss, source/time holdouts, quality, privacy and contamination |
| F1 | Domain-adaptive continuation | OSCR code, OpenWebMath, general-English replay | Specialist uplift without unacceptable general-language forgetting |
| S1 | Response-only instruction SFT | Aya restricted to `eng`, OASST1 reviewed English | Instruction following and factuality regression suite |
| S2 | Task SFT | GSM8K and MBPP approved train examples | Exact math answers and sandboxed code tests |
| P1 | Preference learning | HH-RLHF chosen/rejected pairs | Dedicated objective, human agreement and regression tests |
| V1 | Verified outcome learning | Project-authored English code/math/tool tasks | Independent validators and protected holdouts |
| R1 | Outcome optimisation | Bounded deterministic English tool environments | Matched-budget success, reward-hacking checks and rollback |

Do not use preference pairs as ordinary instruction targets. Do not jump to reward optimisation while the base model cannot reliably follow instructions. A language-model judge must not be the sole verifier of its own output.

## 3. Foundation mixture

The first English foundation mixture uses:
- 80% general FineWeb `sample-10BT` documents.
- 20% FineWeb-Edu `sample-10BT` documents.

This is a document-weight hypothesis, not a token allocation. The pipeline measures actual tokenizer shares. FineWeb-Edu overlaps conceptually and at source level with general Common Crawl web content, so exact and near-duplicate screening is important. The pilot accepts up to 20,000 documents per source; it does not download the full upstream corpus.

Both sources declare ODC-By 1.0 and are derived from Common Crawl. Attribution, upstream terms, privacy, and content filtering require review before model-weight redistribution. Regex checks are not a complete PII audit.

## 4. English-only data gates

- Foundation web text must have an explicit `en`, `eng`, or `english` label and language confidence of at least 0.70. Missing labels and non-English labels are rejected.
- Aya SFT accepts only `language_code=eng`.
- OASST1 accepts only reviewed English messages and keeps entire conversation trees in one split.
- GSM8K and MBPP use only the documented allowed training subsets; protected evaluation tasks stay outside training.
- Every prepared example carries source ID/revision, split, content hash, and relevant licence provenance.
- There are no non-English corpora, text examples, tokenizer targets, or dialogue-evaluation scenarios in the active pipeline.

## 5. Alternatives and lab research

Dolma, DCLM and SlimPajama are candidates for separate matched-budget English experiments, not sources to stack blindly. They are large and may overlap with web material already in FineWeb. Deduplicate and check the precise release's rights and source-component metadata first.

OpenWebMath is an English math-text candidate for domain adaptation, not answer-verified math supervision. OSCR code must be filtered per file/repository licence; there is no universal source-code licence.

Anthropic's public HH-RLHF dataset provides preference pairs, not Claude's private pretraining corpus. DeepSeek's technical reports describe methodology but not a complete downloadable base corpus. DeepSeek-R1 outputs could be considered only as traceable teacher candidates with independent correctness checks and human review. OpenAI does not publish a complete itemised dataset inventory for this ChatGPT model. These disclosures inform independent research; they are not permission to copy private corpora.

## 6. Compute schedule

The GPU notebook requests a Kaggle GPU and uses:
- Foundation: 6,000 optimizer steps, 512-token context, batch size 4, gradient accumulation 4.
- English Aya SFT: up to 10,000 examples and 1,500 steps.
- Task SFT: 1,000 steps.

If no CUDA device is allocated, the notebook explicitly switches to a bounded CPU pilot:
- Foundation: 1,000 steps, 256-token context, batch size 1, FP32.
- English Aya SFT: 250 steps.
- Task SFT: 250 steps.

GitHub Actions validates and orchestrates; it does not supply a GPU. Kaggle availability, account eligibility, accelerator quotas and runtime limits are external constraints. A successful CI run is not evidence of training. Only a completed Kaggle run with a verified checkpoint is a trained model.

## 7. Checkpoint lineage and promotion

Each official foundation checkpoint begins from random initialization in this repository. Later cycles resume only the verified foundation checkpoint and keep the tokenizer byte-identical. Aya- and task-SFT candidates are not fed back into foundation pretraining.

Each cycle exports the exact repository commit, accelerator, step counts, parameter count, checkpoint sizes, source manifests, hashes, held-out evaluations and run lineage. A successful cycle creates a candidate, not an automatically promoted flagship. Promotion requires a reproducible improvement on held-out English tasks without unacceptable regression.
