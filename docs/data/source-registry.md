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
