# Additional candidate data sources and ingestion priorities

Research snapshot: 2026-11-02. Dataset availability and licences can change; every ingestion run must pin an immutable source revision and repeat its licence review.

## Priority 1 — Small, verifiable capability datasets

### GSM8K — mathematics
- Official dataset: https://huggingface.co/datasets/openai/gsm8k
- Licence: MIT, according to its dataset card.
- Main configuration contains 7,473 training examples and 1,319 test examples; the Socratic configuration has the same counts.
- Use only the `train` split for any training experiment. Lock `test` as held-out evaluation material.
- The examples are English grade-school mathematical word problems with worked answers. This is a useful narrow mathematical capability signal, not a substitute for broad scientific or mathematical data.
- The dataset card reports that the answer includes multi-step working. See the official card before using a particular configuration.

### MBPP — basic Python programming
- Official dataset: https://huggingface.co/datasets/google-research-datasets/mbpp
- Reported licence: CC-BY-4.0; retain attribution and re-check the exact current source revision and card.
- The full collection contains 974 basic Python tasks; a sanitized hand-verified subset is smaller (427 tasks in the commonly referenced version).
- Use the official `train` split only. Preserve `validation`, `test`, and `prompt` for protected evaluation/few-shot use. Do not silently merge splits.
- Each task has a natural-language description, reference solution, and test cases. Execute generated code only inside a resource-limited sandbox with no network and a disposable filesystem.

## Priority 2 — Bengali language coverage

### Bangla NLP Corpus v0.2.0
- Dataset: https://huggingface.co/datasets/irfanhossainsust/bangla-nlp-corpus
- Dataset card: CC-BY-SA-4.0; attribution is required and share-alike obligations apply to the released corpus/derivatives. Obtain a project-specific legal review before distributing model weights trained on this corpus.
- The current validation release reports 256,025 documents / about 101.5 million words and identifies the five source collections as Bangla Wikipedia, Wikisource, Wikibooks, Wikiquote, and Wikivoyage. It is not a broad web/news corpus.
- Only the corpus subset is appropriate for general text exposure; task-specific labels are weak/automatic unless the record says human-verified. Do not treat them as ground truth.
- The dataset provides train/validation/test splits and extensive record provenance. Do not combine validation/test records into training.
- Content is strongly encyclopedic and is not enough by itself for colloquial, conversational Bangla or Banglish.


### B-CORE — large Bangla pretraining candidate

- Dataset card: https://huggingface.co/datasets/nahid-hub/B-CORE-bengali-corpus
- The current card reports approximately 16.47 million documents, 4.32 billion tokens, and 20.8 GB compressed. It advertises streaming access.
- Dataset-level licence is stated as CC-BY-4.0, but the card explicitly notes that individual source components may carry their own attribution requirements. Review the source-data manifest and component licences before ingestion or release; do not treat the top-level label as blanket clearance for every underlying work.
- This is a promising Bangla monolingual foundation source, but it does not replace colloquial Banglish, dialogue, code-switching, or English. Keep its contribution separately measured and avoid letting it dominate the multilingual mixture.
- Its full compressed size is not suitable for our present free CPU CI workflow. A small streaming sample can test the loader, tokenizer fertility, source metadata, and PII checks; it is not a meaningful full-scale pretraining run.
- Retain citation/attribution, immutable dataset revision, document IDs, source metadata, and hashes. Keep any source-level exclusions auditable.

### Bangladesh National Corpus
- Landing page: https://www.bangladeshnationalcorpus.org/
- Large reported collection and useful potential source, but this research pass did not establish a clear reuse/training licence for all materials.
- **Do not ingest** until dataset-level and component-level permissions are confirmed.

## Priority 3 — Licensed source-code material with provenance

### OpenScientificCodeRegistry (OSCR)
- Data access: https://openscicode.org/help/data/
- The scripts dataset contains source files only when the original repository licence was verified as allowing redistribution. Per-repository manifests preserve source URL, commit, licence, and per-file hashes.
- The dataset does not replace each script's original licence. Filter to a clearly permitted licence set, preserve attribution/provenance, and exclude non-commercial sources from any commercial training/deployment plan.
- Suitable for a small, carefully selected Python slice; not a replacement for broad code pretraining.

### The Stack (BigCode)
- Project documentation: https://www.bigcode-project.org/docs/about/the-stack/
- Multi-terabyte corpus spanning hundreds of languages; far too large for free CPU CI.
- Each file retains its original repository licence and attribution conditions. The maintainers require users to track the newest allowed release because opt-out removals are processed over time.
- Do not download or ingest the whole dataset in our current environment. A future small filtered subset would need a current source release, opt-out compliance, original licence metadata, and file-level provenance.

## Current training plan

1. Keep OASST1 as the first conversational source. It is an older 2023 sample, not current and not sufficient for Bengali.
2. Add a small, train-split-only GSM8K and MBPP branch after response-only masking and a task-aware evaluation harness exist.
3. Add Bengali corpus content only after the ShareAlike implications for downstream weight distribution are reviewed; use it as a separate language mixture and measure Bengali tokenisation separately.
4. Investigate OSCR for carefully filtered, attribution-preserving code examples.
5. Do not use The Stack or unlicensed public web scraping as a shortcut.

## Global exclusions and quality gates

- Do not scrape private repositories, paywalled pages, private social posts, or content behind access controls.
- Do not ingest credentials, secrets, personal records, or private user conversations.
- Preserve source, immutable revision, licence, attribution, hash, split, preprocessing version, and exclusion reason.
- Deduplicate exact and near-duplicate content across sources before mixture creation.
- Keep benchmark test/prompt splits untouched; report potential contamination.
- Validate code with offline, resource-limited tests; never execute arbitrary corpus code on the host.


## Additional foundation-pretraining candidate

### FineWeb and FineWeb-Edu

- FineWeb dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb is an English web-text corpus derived from 96 Common Crawl dumps from 2013 through April 2024. The current dataset card reports more than 18.5 trillion tokens and the ODC-By 1.0 licence.
- FineWeb-Edu is an educationally filtered subset reported at roughly 1.3 trillion tokens and ODC-By 1.0. Common Crawl terms also apply.
- These are foundation-pretraining candidates, not instruction-tuning datasets. Use a pinned release/configuration, retain source metadata, filter duplicates and quality issues, and create source/temporal held-out evaluation.
- **Scale caveat:** the complete corpora are far beyond our current free CPU environment. FineWeb offers a `sample-10BT` configuration (about 27.6 GB); streaming a small, reproducible document sample could validate ingestion and tokenizer statistics, but it is not meaningful frontier pretraining and must not be presented as equivalent to corpus-scale training.
- Retain the legal/provenance record. ODC-By attribution and Common Crawl terms must be respected.

- **Privacy caveat:** the FineWeb card says email addresses and public IP addresses are anonymized, but residual personally identifiable information is still likely to be present; phone numbers are not regex-filtered due to false-positive concerns. Run our own privacy/PII screening and exclusion pipeline before ingestion. The card also notes that some toxic or harmful material remains and that code is not prevalent. Do not treat the upstream filters as a guarantee.

## Additional foundation and specialist pretraining candidates

### DCLM-Baseline — curated general-English pretraining
- Dataset card: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- The card reports CC-BY-4.0 and describes a research baseline derived from Common Crawl, with a much larger DCLM-Pool behind it.
- The dataset is explicitly described as a research baseline, not intended by its authors as a production-ready corpus or as a specialist code/math source.
- The underlying Common Crawl source terms and original-content rights still need review. Retain the dataset card, revision, attribution, source metadata, and any upstream restrictions. Do not equate a collection-level CC-BY tag with blanket clearance for every original webpage.

### Dolma — diverse English pretraining corpus
- Dataset: https://huggingface.co/datasets/allenai/dolma
- The published corpus is approximately 3 trillion tokens from web, academic publications, code, books, and encyclopedic sources. Dolma v1.6 has a sample release described as roughly 10B tokens and 16.4 GB compressed.
- Dataset-level licence: ODC-BY. The dataset card explicitly says original source licences and terms of use still apply. The mixture includes sources such as Common Crawl, The Stack, Reddit, and academic content; review component-level permissions and opt-out state before selecting any subset.
- The Dolma curation toolkit is separate from the dataset and is Apache-2.0. Reusing the toolkit does not grant additional rights to the corpus.

### OpenWebMath — mathematics-domain continuation
- Dataset: https://huggingface.co/datasets/open-web-math/open-web-math
- The current card reports 6.3 million rows and about 27.4 GB. It focuses on English mathematical web text and LaTeX extraction.
- Dataset-level licence: ODC-BY 1.0; Common Crawl terms apply, and the project states it does not change the licences of underlying data.
- Best considered for a math-heavy continued-pretraining mixture after broad foundation pretraining—not as a replacement for verified arithmetic, theorem, or code-execution tasks.


## Capability-specific source decisions (2026-11-03)

### Math instruction corpora

- **OpenMathInstruct-2**: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
- Its Hugging Face card currently declares CC-BY-4.0 and reports about 14 million problem/solution pairs. The card says the solutions were generated using Llama 3.1 405B Instruct.
- **Artemis decision: do not ingest by default.** Even though this is a dataset rather than imported weights, its externally model-generated solution traces create a distillation/provenance dependency that does not fit our strongest independent-training preference. Reconsider only if the project explicitly approves model-generated data and audits the dataset card, source problems, and attribution obligations.

### Science and commonsense evaluation

- AI2 ARC: https://huggingface.co/datasets/allenai/ai2_arc
- The benchmark is useful for grade-school science QA and has public train/validation/test splits, but its current metadata says CC-BY-SA-4.0 while the dataset card's licensing section has been reported as incomplete/“More Information Needed”. Treat it as **evaluation-only / licence review required** for now; do not ingest its training split until the underlying rights and ShareAlike implications are verified.
- Publicly accessible data is not automatically reusable training data. Keep benchmark splits protected even when the underlying dataset is downloadable.

### General pretraining

- FineWeb and FineWeb-Edu remain the highest-priority broad English pretraining candidates for future accelerator-backed training. Both are labelled ODC-By on their Hugging Face dataset cards. FineWeb is sourced from Common Crawl and its card explicitly acknowledges that residual PII may remain despite email/IP anonymisation. Apply Artemis's own PII and provenance filters, retain attribution, and review Common Crawl terms before use.
- B-CORE remains the most promising identified large Bangla-only candidate: its card reports 16.5M documents and about 4.32B tokens under a top-level CC-BY-4.0 label. The card notes that underlying components may have separate attribution/rights requirements. Audit its source manifest before any download or use.

