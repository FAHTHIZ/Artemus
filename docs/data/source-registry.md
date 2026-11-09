# English-only training data source registry

**Scope:** ArtemisLM training accepts English text only. The ingestion pipeline enforces an English language label and a minimum language-confidence score for foundation corpora; Aya instruction data is restricted to the ISO 639-2 code `eng`. Non-English corpus sources and evaluation examples are excluded.

## Active foundation sources

| Source | Role | Declared licence / terms | Pipeline gate |
|---|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT` | Broad English web pretraining | ODC-By 1.0; Common Crawl terms and source provenance apply | Resolve immutable revision at run start, English label, language score ≥0.70, privacy/quality filters, deduplication |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` | Educational English supplement | ODC-By 1.0; Common Crawl terms apply | Pinned revision, English label, language score ≥0.70, deduplicate against FineWeb |

The initial document-weight hypothesis is 80% FineWeb and 20% FineWeb-Edu. Those are **document** quotas, not token quotas. The tokenizer must measure actual source token shares. The complete upstream datasets are much larger than the initial pilot; a configured sample is not a claim that every upstream token was downloaded or trained.

## English instruction and task sources

| Source | Role | Declared licence / terms | Rules |
|---|---|---|---|
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Human-annotated instruction SFT | Apache-2.0 per upstream card | Retain only `language_code=eng`; discard annotator IDs and demographics; group identical prompts into one split |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | English human assistant conversations | Apache-2.0 per upstream card | Reviewed English messages only; deleted/synthetic/positive-PII records excluded; conversation trees stay in one split |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | English math word problems | MIT | Official train only; preserve official test for evaluation |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | English natural-language-to-Python tasks | CC-BY-4.0 per upstream metadata | Approved task IDs only; preserve attribution; execute candidates in a network-disabled sandbox |

## Specialist and alternative English sources

- **[OSCR](https://huggingface.co/datasets/OpenScientificCodeRegistry/Database):** optional code continuation. There is no single licence for every file; use exact per-repository/per-file SPDX evidence and allowlisted licences only. Preserve original repo, commit, file path and hash.
- **[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math):** English mathematical web documents and formulas. ODC-By 1.0 plus underlying source terms. It is pretraining text, not verified math answers; keep it in a separate domain-adaptation stage.
- **[Dolma](https://huggingface.co/datasets/allenai/dolma):** open large-scale pretraining corpus research from AI2. Evaluate as an alternative source/release after source-component, licence, privacy and duplication review.
- **[DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0):** large quality-filtered web corpus. Multi-terabyte; assess storage/compute and exact licence. Do not stack with FineWeb without cross-source deduplication.
- **[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B):** smaller open pretraining alternative. Review constituent-source terms and deduplicate against current data before a matched-budget comparison.

## Research by leading labs: what is and is not public

- **Anthropic:** Claude transparency materials describe broad categories of proprietary sources, but do not expose a complete downloadable Claude pretraining corpus. [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) is a public helpfulness/harmlessness preference-pair dataset under MIT. It is **not** ordinary dialogue SFT; use only after implementing a dedicated preference/reward-model objective and reviewing the dataset card's cautions. Red-team preference data must not be blindly mixed into supervised dialogue data.
- **DeepSeek:** public technical reports describe broad use of public and licensed material, data filtering and post-training methods, not a complete itemised base pretraining corpus. DeepSeek-R1's published model/code terms and output examples may inform a later teacher-candidate experiment. Any teacher outputs must be traceable to an exact model/version/terms and independently verified and human-reviewed before training.
- **OpenAI:** public research and product documentation provide methodological context and examples of dataset families, but not a complete itemised list of the exact data used to train this ChatGPT model.
- **AI2/OLMo:** openly released datasets such as Dolma are useful for auditable independent research; their availability does not eliminate component-level rights and privacy checks.

**Do not claim to borrow Anthropic's or DeepSeek's private base-model corpus.** A model report is not a data download, and publicly visible does not automatically mean licensed for training or redistribution.

## Global acceptance gates

1. Pin immutable revisions and record retrieval time, URL, licence, upstream terms and review note.
2. Enforce English language metadata and confidence; reject missing metadata and every non-English label.
3. Sample and inspect content before training; record quality and privacy rejection counts.
4. Deduplicate within and across sources; group related conversations, prompts and repository files to prevent split leakage.
5. Preserve per-record source IDs, hashes, transformation versions and final token counts.
6. Keep benchmark tests out of training and tokenizer fitting.
7. Never ingest private conversations, private repositories, paywalled sources, credentials, secrets, opt-out content or material with unclear rights.
8. Model-generated text remains candidate-only until independently checked and human-reviewed.
