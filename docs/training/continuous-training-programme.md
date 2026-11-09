# Artemus English-only continuous training programme

**Status:** Active training plan  
**Language scope:** English only  
**Purpose:** build a reproducible, source-audited training pipeline and measure capability per compute budget.

## 1. Current checkpoint

Previously documented compact checkpoints are early random-initialised task-SFT experiments on OASST1, GSM8K and MBPP. They are useful for validating the trainer and inference interfaces, but recorded mathematics and Python generations remain incoherent. They are not broad foundation-pretrained models and must not be represented as useful general assistants.

The next official foundation run starts from fresh random initialization with a tokenizer trained only on the approved English foundation mixture.

## 2. Training stages

| Stage | Method | English data | Promotion gate |
|---|---|---|---|
| F0 | Foundation next-token pretraining | FineWeb + FineWeb-Edu | Held-out loss, source/time holdouts, quality and privacy |
| F1 | Domain-adaptive continuation | OSCR code, OpenWebMath, general-English replay | Specialist uplift without general-language forgetting |
| S1 | Response-only instruction SFT | Aya restricted to `eng`, reviewed OASST1 English | Instruction following and factuality regression suite |
| S2 | Task SFT | GSM8K and MBPP approved train examples | Exact math answers and sandboxed code tests |
| P1 | Preference learning | HH-RLHF chosen/rejected pairs | Dedicated objective, human agreement and regression |
| V1 | Verified outcome learning | Project-authored English code/math/tool tasks | Independent validators and protected holdouts |
| R1 | Outcome optimisation | Bounded deterministic tool environments | Matched-budget success, reward-hacking checks and rollback |

Do not use preference pairs as ordinary instruction targets. Do not jump to reward optimisation while the base model cannot reliably follow instructions. A language-model judge must not be the sole verifier of its own output.

## 3. Foundation mixture

The first English foundation mixture uses 80% general FineWeb `sample-10BT` documents and 20% FineWeb-Edu `sample-10BT` documents. This is a document-weight hypothesis, not a token allocation. The pipeline measures actual tokenizer shares. FineWeb-Edu may overlap with general web content, so cross-source deduplication is required. The pilot accepts up to 20,000 documents per source; it does not download the complete upstream corpora.

Both sources declare ODC-By 1.0 and derive from Common Crawl. Attribution, underlying terms, privacy and content filtering require review before model-weight redistribution. Regex checks are not a complete PII audit.

## 4. English-only data gates

- Foundation web text must have an explicit `en`, `eng` or `english` label and language confidence of at least 0.70. Missing labels and other language labels are rejected.
- Aya SFT accepts only `language_code=eng`.
- OASST1 accepts only reviewed English messages and keeps entire conversation trees in one split.
- GSM8K and MBPP use only approved training subsets; protected evaluation tasks stay outside training.
- Prepared examples carry source IDs/revisions, splits, content hashes and licence provenance.
- The active pipeline contains no non-English corpora, tokenizer targets or dialogue-evaluation examples.

## 5. Alternatives and leading-lab research

Dolma, DCLM and SlimPajama are candidates for separate matched-budget English experiments, not sources to stack blindly. They are large and may overlap with web material already in FineWeb. Deduplicate and review precise release rights first.

OpenWebMath is an English math-text candidate for domain adaptation, not verified answer supervision. OSCR code must be filtered per-file and per-repository licence; there is no universal licence for the full collection.

Anthropic's public HH-RLHF dataset supplies preference pairs, not Claude's private pretraining corpus. DeepSeek technical reports describe methodology but not a full downloadable base corpus. DeepSeek-R1 outputs may be considered only as traceable teacher candidates with independent correctness checks and human review. OpenAI does not publish a complete itemised dataset inventory for ChatGPT. These disclosures inform independent research; they are not permission to copy private corpora.

## 6. Compute schedule

The Kaggle GPU notebook requests a GPU and uses 6,000 foundation steps (512-token context, batch 4, gradient accumulation 4), up to 10,000 English Aya examples for 1,500 steps, and 1,000 task-SFT steps.

If no CUDA device is allocated, it explicitly switches to a bounded CPU pilot: 1,000 foundation steps (256-token context, batch 1, FP32), 250 English Aya steps and 250 task-SFT steps.

GitHub Actions validates and orchestrates; it does not supply a training GPU. Kaggle availability, account eligibility, accelerator quotas and runtime limits are external constraints. A successful CI run is not evidence of training. Only a completed Kaggle run with a verified checkpoint constitutes a trained model.

## 7. Checkpoint lineage and promotion

Each official foundation checkpoint starts from random initialization in this repository. Later cycles resume only the verified foundation checkpoint and keep the tokenizer byte-identical. Aya- and task-SFT candidates are never fed back into foundation pretraining.

Each cycle exports the exact repository commit, accelerator, step counts, parameter count, checkpoint sizes, source manifests, hashes, held-out evaluations and run lineage. A successful cycle creates a candidate, not an automatically promoted flagship. Promotion requires reproducible improvement on held-out English tasks without unacceptable regression.
