# English-only training data source registry

ArtemusLM accepts English text only. Foundation corpora require an explicit English label and confidence score ≥0.70. Aya instruction data is filtered to ISO code `eng`. No non-English corpora or evaluation samples belong in the active pipeline.

## Active foundation sources

| Source | Role | Licence / terms | Gate |
|---|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT` | General English web pretraining | ODC-By 1.0; Common Crawl terms apply | Immutable revision, English label, score ≥0.70, privacy/quality filters and deduplication |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` | Educational English supplement | ODC-By 1.0; Common Crawl terms apply | Pinned revision, English label, score ≥0.70, deduplicate against FineWeb |

Initial document-weight hypothesis: 80% FineWeb / 20% FineWeb-Edu. These are document quotas, not token quotas. Measure actual shares with the Artemis tokenizer. A bounded pilot does not imply full upstream ingestion.

## English instruction and task data

| Source | Role | Declared licence | Rules |
|---|---|---|---|
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Human-annotated instruction SFT | Apache-2.0 per upstream card | `language_code=eng` only; discard annotator IDs/demographics; group identical prompts across splits |
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Reviewed assistant conversations | Apache-2.0 per upstream card | Reviewed English messages only; exclude deleted, synthetic and positive-PII records; conversation-tree split |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | English math word problems | MIT | Official train only; official test protected |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | English natural-language-to-Python | CC-BY-4.0 per upstream metadata | Approved IDs only; preserve attribution; sandbox generated code |

## Specialist and alternative English sources

- **[OSCR](https://huggingface.co/datasets/OpenScientificCodeRegistry/Database):** optional code continuation. There is no single licence for every file; preserve exact repository/file SPDX evidence, commits, paths and hashes. Allowlist reviewed licences only.
- **[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math):** English mathematical web documents and formulas. ODC-By 1.0 plus underlying source terms. Pretraining text only, not verified answer supervision.
- **[Dolma](https://huggingface.co/datasets/allenai/dolma):** large open pretraining corpus research from AI2. Evaluate a precise release after component-source, licence, privacy and duplication review.
- **[DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0):** quality-filtered web corpus; multi-terabyte scale. Review the exact release and terms; do not stack blindly with FineWeb.
- **[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B):** smaller open pretraining alternative. Review constituent licences and deduplicate before matched-budget comparisons.

## Leading-lab research: public versus private

- **Anthropic:** public transparency materials describe broad source categories, not a downloadable complete Claude pretraining corpus. [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) is a separate MIT preference-pair dataset, not ordinary dialogue SFT; defer until a dedicated preference/reward objective exists and review the upstream cautions.
- **DeepSeek:** public technical reports describe broad public/licensed data and quality methods, not a complete itemised base corpus. DeepSeek-R1 release terms and outputs may inform future teacher candidates; require exact model/version provenance, independent verification and human review.
- **OpenAI:** public research and product documentation provide methodological context and selected examples, not a complete itemised training-data inventory for this ChatGPT model.
- **AI2/OLMo:** open releases such as Dolma support auditable independent research but do not waive source-level rights and privacy review.

Do not claim to borrow private lab corpora. A research report is not a dataset download; public visibility is not automatic permission to train or redistribute.

## Acceptance gates

1. Pin immutable revisions, retrieval time, URLs, licences, source terms and review notes.
2. Reject missing language metadata and non-English labels before tokenizer fitting.
3. Inspect samples and record quality/privacy rejection counts.
4. Deduplicate within and across sources; group prompts, conversations and repositories across splits.
5. Preserve source IDs, hashes, transformation versions and final token counts.
6. Keep benchmark test sets out of training and tokenizer fitting.
7. Exclude private conversations/repositories, paywalled sources, credentials, secrets, opt-out content and unclear-rights material.
8. Keep model-generated text candidate-only until independently verified and human-reviewed.
