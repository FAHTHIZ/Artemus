# Initial data-source registry

This registry records candidate data sources before they enter a training run. Publicly reachable does not automatically mean unrestricted. The current pipeline pins versions, checks hashes, records licences, filters known quality flags, and keeps train/validation conversation trees separate.

## In the first reproducible run

### OpenAssistant Conversations (OASST1)

- **Use:** human-generated assistant-style conversations for a first language-modeling experiment.
- **Dataset card:** https://huggingface.co/datasets/OpenAssistant/oasst1
- **Pinned source revision:** `a4992e1595393714c9e6c567b1ed645bbf7d00ee`
- **Selected export:** `2023-04-12_oasst_ready.messages.jsonl.gz` (88,838 messages in upstream metadata).
- **Dataset card licence:** Apache-2.0. Review the included upstream licence and attribution conditions before redistributing derivatives.
- **Filters:** English only; review_result=true; deleted=false; synthetic=false; positive PII labels excluded.
- **Integrity:** expected SHA-256 is pinned in `src/artemis/prepare_oasst1.py`.
- **Split:** stable hash of message_tree_id, so a conversation tree cannot leak across train and validation.
- **Limits:** this is a 2023 dataset and not a complete or current representation of assistant behaviour. The baseline byte tokenizer and tiny network are inadequate for strong instruction following. Dataset labels do not guarantee that every retained message is error-free.

Do not commit the raw archive, processed corpus, or trained checkpoint to Git. The workflow downloads and processes the archive during the run and retains the run artifact only temporarily.

## Candidate next sources (not yet ingested)

### GSM8K

- **Use:** mathematical word-problem examples; candidate for a separately tracked math training subset.
- **Dataset card:** https://huggingface.co/datasets/openai/gsm8k
- **Dataset card licence:** MIT.
- **Important:** preserve the official train/test split. Use the training split for training only; keep test data exclusively for evaluation. The present model is too small to claim reasoning capability from a short exposure.

### MBPP

- **Use:** basic natural-language-to-Python tasks with reference solutions and tests.
- **Dataset card:** https://huggingface.co/datasets/google-research-datasets/mbpp
- **Reported licence:** CC-BY-4.0; re-check the current upstream card and terms before ingestion.
- **Important:** preserve official splits and reserve the test split for evaluation. Run every candidate solution through tests; do not treat text likelihood as code correctness.

### The Stack (BigCode)

- **Use:** possible future source-code pretraining after infrastructure and licence review.
- **Project documentation:** https://www.bigcode-project.org/docs/about/the-stack/
- **Scale:** multi-terabyte corpus; not suitable for the first CPU run.
- **Conditions:** repository-level original licences and attribution obligations continue to apply; the dataset has its own terms and opt-out governance. Obtain and follow the current dataset terms and use the most recent permitted release. Do not simply scrape GitHub or assume a public repository grants blanket training rights.

## Data-quality rules

1. Record a source URL, immutable revision, licence, retrieval timestamp, and content hash.
2. Keep source-specific filters and transformations in version control.
3. Exclude or quarantine secrets, credentials, private information, and data with unclear rights.
4. Deduplicate before mixing sources; preserve source IDs for traceability.
5. Keep evaluation data out of training and report contamination risks.
6. Prefer execution-verified code examples over untested snippets.

## Current training pipeline and reviewed source expansion

*Reviewed 2026-11-07. This section records what is already wired into the repository versus what still needs a separate ingestion and rights review. A dataset card's top-level licence is not proof that every underlying source has identical terms.*

### Integrated into the GPU notebook

| Source | Current use | Licence / rights notes | Gate |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` | Bounded English foundation-pretraining stream; notebook accepts at most 20,000 documents | Dataset card declares ODC-By 1.0. The source derives from Common Crawl, whose terms also apply. Internet-derived content can still contain residual personal information. | Research pilot only; inspect samples, preserve attribution and provenance, do not assume regex filters make the corpus PII-free. |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), `ben_Beng` | Bounded Bengali foundation stream; notebook accepts at most 5,000 documents and measures the actual token share after tokenization | Dataset card declares ODC-By 1.0 and describes a Common Crawl-derived multilingual corpus. The card reports 15,185,742 Bengali documents, 6,153,579,265 words and 22.25 GB. Email and public-IP anonymization are documented; phone numbers are not filtered because of false-positive risk. | Keep bounded; add local privacy screening and sample review. Word counts are not tokenizer token counts. |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Multilingual, response-only instruction SFT; notebook caps the sample at 10,000 examples | Dataset card declares Apache-2.0 and reports about 204,000 human-annotated prompt/completion pairs across 65 languages (71 including dialects and scripts). The source also exposes annotator-related fields. | Use only the approved prompt/target and language/annotation fields; exclude annotator IDs and demographics. Human annotation is not a factuality guarantee. |
| [OSCR Database](https://huggingface.co/datasets/OpenScientificCodeRegistry/Database) | Optional Python code continued-pretraining stage | There is no single licence for every code file. Each repository manifest retains its source repository, commit, licence, confirmation evidence, file paths and hashes. | The notebook's `OSCR_MANIFEST_PATHS` is empty by default, so this stage is skipped until exact, reviewed repository manifests are selected. |

### Used by the recurring GPU task-SFT stage (and previously available in the CPU experiment)

| Source | Role | Licence / split discipline |
|---|---|---|
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Human-generated assistant conversations | Dataset card declares Apache-2.0. The current pipeline pins a ready export and groups complete conversation trees when splitting. The current preparation path is English-focused. |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Arithmetic word-problem supervision and evaluation | MIT. Train on the official training split only; preserve the official test split for evaluation. |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | Natural-language-to-Python task supervision | Upstream dataset metadata reports CC-BY-4.0. Retain attribution and the repository's explicitly selected train/validation/test ID boundaries; validate candidate code with tests in an isolated runner. |

### High-value candidates for a separate ingestion stage

| Candidate | Why it is useful | Licence and caution | Recommended status |
|---|---|---|---|
| [B-CORE Bengali Corpus](https://huggingface.co/datasets/nahid-hub/B-CORE-bengali-corpus) | Large Bengali text collection; current card reports about 16.47 million documents and 20.8 GB | The dataset card declares CC-BY-4.0 but explicitly warns that individual source components may impose their own attribution requirements. | High-priority review candidate. Build a source-component rights map before downloading into the training pipeline. |
| [Bangla NLP Corpus](https://huggingface.co/datasets/irfanhossainsust/bangla-nlp-corpus) | Smaller, provenance-rich Bengali corpus; card reports 256,025 documents and about 101.5 million words, with source URLs, license fields and Bangladesh-specific PII detectors | CC-BY-SA-4.0. The current release is a validation release built from Wikimedia-family sources; task labels are weak/automatic unless marked human-verified. | Useful controlled pilot after review of share-alike obligations; do not treat its derived weak labels as ground truth. |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Mathematical web text and formula-rich material for domain-adaptive pretraining | ODC-By 1.0; Common Crawl terms and original source terms still apply. It is not a set of verified math answers. | Add only as a continued-pretraining/replay source, separate from GSM8K-style answer supervision. |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | Bengali speech/audio-text examples for a future speech-perception branch | Dataset card reports CC-BY-4.0; the Bengali subset is audio data, not ordinary text. | Keep separate from the current text-only language-model corpus until an audio pipeline exists. |

### Do not ingest by default

- Do not scrape arbitrary public Git repositories and assume that public visibility grants training or redistribution rights.
- Do not ingest large mixed code collections until repository-level licences, provenance, attribution and any opt-out/access conditions are verifiable. Prefer OSCR's exact manifest path and licence evidence.
- Do not train on private conversations, credentials, personal records, or unreviewed model-generated answers.
- Do not mix speech/audio, weak labels, web math text, and assistant-response examples into one undifferentiated text stream. They need separate objectives, validation sets and training phases.

### Acceptance gates for every new source

1. Pin an immutable upstream revision and record retrieval time, source URL, declared licence, underlying source terms and any permission evidence.
2. Sample records before ingestion. Measure language correctness, document quality, duplicate rate, boilerplate, contamination and personal-information leakage. A dataset's own filtering is not a complete privacy audit.
3. Deduplicate within and across sources. Group splits by conversation, prompt, document/source, repository or other shared origin so near-identical records cannot leak between training and validation.
4. Preserve per-record provenance, hashes, transformations, language and task labels. Record the final token share after applying the actual Artemis tokenizer.
5. Keep benchmark test sets outside every training and model-selection mixture. For code and math, score executable or exact outcomes, not just next-token loss.
6. Promote a candidate checkpoint only after it improves held-out task scores without regressions on protected language, reasoning, code, memory and safety tests. Keep the previous checkpoint for rollback.

A corpus can be large, filtered and useful without being error-free, rights-cleared for every downstream use, or appropriate for every training objective. The goal is traceable, measurable data—not maximum row count.

### Per-cycle task-SFT use

The recurring GPU notebook now consumes OASST1, GSM8K and MBPP after Aya SFT. This is a distinct task-SFT phase, not foundation pretraining. OASST1 is sampled with a bounded per-cycle example cap; GSM8K and MBPP remain pinned to the source revisions and task boundaries described above. The final task checkpoint is not used as the next foundation checkpoint. See `data/training-curriculum.json` for the machine-readable cycle definition.

