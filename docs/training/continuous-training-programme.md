# Artemus continuous training programme

**Status:** Active training plan  
**Last reviewed:** 2026-11-06  
**Purpose:** Turn independent model training into a staged, source-audited, GPU-capable programme.

## 1. What the current checkpoint actually is

The most recent documented 1,004,928-parameter checkpoint is an early task-supervised experiment trained from random initialization on OASST1, GSM8K, and MBPP. It reached 16,000 optimizer steps in a later run, but the run's multi-turn evaluation failed because a Bangla follow-up prompt exceeded the 256-token context. The checkpoint artifact was uploaded despite that evaluation failure.

That checkpoint is useful as a reproducible experiment and for testing inference/teaching interfaces. It is **not** a broad foundation-pretrained model, and it should not be relabelled as one. The first general pretraining run below starts from random initialization with a separately trained tokenizer.

## 2. The training ladder

| Stage | Method | Data / feedback | Promotion gate |
|---|---|---|---|
| F0 | Foundation pretraining | Broad, pinned, deduplicated general text; document-level EOS | Held-out next-token loss, source and temporal holdouts, contamination checks |
| F1 | Continued/domain-adaptive pretraining | Code, math, science, Bangla/multilingual corpora under a bounded mixture | Improves target domain without unacceptable general-language forgetting |
| S1 | Instruction SFT | Reviewed human-authored instruction/conversation pairs, OASST1, corrected examples | Protected instruction-following and multilingual evaluation |
| S2 | Task-specific SFT | Official training splits only: GSM8K, MBPP, curated code tasks | Exact-answer or test-based evaluation on held-out tasks |
| V1 | Verified outcome training | Code solutions that compile/pass tests; math with exact checks; structured API outputs | Independent verifier, no sandbox escape, private holdouts |
| P1 | Preference learning | Human chosen/rejected responses; explicit direct-teaching feedback | Human preference agreement and regression suite |
| R1 | Environment/outcome optimisation | Tool-use tasks in deterministic or sandboxed environments; verifiable reward | Matched-budget success rate, reward-hacking checks, rollback checkpoint |
| M1 | Multimodal extensions | Licensed image/audio/video/text pairs and human-reviewed annotation | Modality-specific held-out tests and provenance audits |

Do not jump to preference/RL training while the base model cannot reliably follow instructions. Do not use a language-model judge as the only verifier for its own answers.

## 3. Data source plan

### Start now: small, reproducible pilots

- **FineWeb-Edu sample-10BT** — broad educational English text for an initial foundation-pretraining pilot. Dataset revision pinned in the GPU notebook to 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9; the card declares ODC-By 1.0. Common Crawl source terms and residual PII caveats remain applicable. The notebook samples a bounded number of documents; it does not download 10 billion tokens.
- **OSCR Python** — code-domain continuation through selected repository manifests and referenced Parquet blocks only. Dataset revision pinned to b0e3fadb276678880f7f3d764308fbebd3bcd044. Each repository/file keeps its own licence and source commit; exact-SPDX allowlisting and per-file SHA verification are mandatory.
- **OASST1** — already pinned to a4992e1595393714c9e6c567b1ed645bbf7d00ee, Apache-2.0 per its dataset card; retain the existing tree-grouped split and quality filters.
- **GSM8K** — use official train only for math SFT; keep test locked for evaluation. Pinned data revision and hash are in prepare_capability_data.py.
- **MBPP** — use allowed training IDs only; retain official protected test/prompt ranges and execute generated code in an isolated offline sandbox.

### Add after source/legal review

- **FineWeb / FineWeb2** — broader and multilingual web pretraining. Keep the Common Crawl provenance, attribution, privacy filtering, and per-language sample balance. FineWeb-Edu is the first bounded pilot, not an excuse to dump all Common Crawl data into a training run.
- **B-CORE** — important Bangla pretraining candidate. Its card declares CC-BY-4.0, but the underlying component/source rights need their own audit before large-scale use or weight distribution. Keep it separate from the initial English mix.
- **Bangla NLP Corpus** — useful for Bangla language coverage but its CC-BY-SA terms require a distribution-policy review. Do not fold it into default training without that review.
- **OpenWebMath** — mathematics continued-pretraining candidate, subject to the upstream source/attribution review noted in the data registry.
- **DCLM-Baseline and Dolma** — broader general-text alternatives; review original-source terms, attribution, privacy, and dataset-specific restrictions before adding.
- **The Stack v2** — do not bulk-download now. Access agreements, Software Heritage provenance, original licences, removal/opt-out updates, and model-release terms are prerequisites.
- **CodeSearchNet** — do not ingest until exact per-file rights are resolved; it may still be useful as a retrieval/evaluation dataset.

### Do not ingest

Private conversations without explicit consent, private repositories, paywalled or access-controlled sources, credential/secret dumps, opt-out data, unlicensed books/code, or a dataset whose rights cannot be determined. Public availability alone is not a licence.

## 4. Mixture strategy

Do not let a single enormous source dominate all training. The first general-text pilot is intentionally one source so the pipeline is measurable. The next curriculum should use a manifest-driven mixture with a target token budget per source and language.

Suggested later starting mixture (a hypothesis, not a fixed recipe):
- 55–65% high-quality general/educational text.
- 10–20% technical/scientific text.
- 10–15% code with verified source licences.
- 5–10% mathematics.
- 5–10% Bangla/multilingual material, adjusted based on tokenisation and evaluation.

For every source and stage, record the actual token counts, not just document counts. Reweight from measured tokenizer fertility and held-out capability results. Preserve a representative general replay fraction during domain adaptation to reduce catastrophic forgetting.

## 5. Human teaching: let people talk to the model

The new artemus-teach command gives a direct terminal conversation loop:

    artemus-teach --checkpoint artifacts/artemus-foundation.pt

For each response, the teacher can correct it, explicitly approve it, or skip it. Corrections are written to a candidate JSONL dataset with session ID, checkpoint hash, timestamp, and source type. Model-generated answers are **not** accepted by default. Enabling --allow-human-approved-model-output is a deliberate opt-in to human-approved self-distillation.

Teaching data is not training data until it is reviewed and selected for a later SFT run. This avoids silently converting model hallucinations or private conversation material into weight updates.

## 6. Other learning channels

- **Verified code training:** pair each prompt with a reference implementation and offline tests; use execution results as independent correctness evidence.
- **Math outcome training:** evaluate final answers exactly where possible; retain derivations for learning but do not score a fluent derivation as proof.
- **Preference optimisation:** collect human comparisons of two candidate answers with task context and a reason; DPO-like training comes only after enough high-quality, diverse pairs exist.
- **Reward/outcome optimisation:** use a deterministic task environment with explicit success conditions and bounded actions; never reward length, confidence, or self-reported completion.
- **Synthetic examples:** generate candidates from a teacher only when explicitly allowed by project policy; preserve teacher/source metadata, filter with independent checks, and label them as synthetic.
- **Self-play:** useful later for tasks with objective rules (games, code tests, simulators); not a replacement for real-world language coverage.
- **Retrieval and memory:** improve the integrated Artemus/OASIS system without altering weights. Measure retrieval/tool gains separately from base-model gains.
- **Human written demonstrations:** strongest direct teaching channel available now; prioritise corrected examples over raw thumbs-up approvals.

## 7. Run control and evaluation

Each training segment must:
1. Pin the code commit, data revision, data manifest, tokenizer hash, and parent checkpoint hash.
2. Use a named phase and explicit token/step/compute budget.
3. Save a recoverable checkpoint at evaluation intervals.
4. Keep source/time/language/group holdouts untouched.
5. Compare against the previous checkpoint on identical tasks, prompts, and budgets.
6. Report train/validation loss and actual tokens processed, then run capability tests.
7. Stop on divergence, data leakage, repeated validation regression, or a failed integrity gate.
8. Promote only a checkpoint with a complete provenance record and a passing regression suite.

“Continuous training” means a repeatable curriculum that advances to new reviewed data and validated objectives. Repeating one small dataset endlessly is not continuous improvement; it is overfitting.

## 8. First runnable route

Open notebooks/kaggle-gpu-training.ipynb in Kaggle, enable a GPU, and run the notebook top to bottom. It:
- checks GPU allocation;
- clones and records the Artemus commit;
- streams a bounded FineWeb-Edu sample at a pinned dataset revision;
- filters obvious PII/secret patterns, duplicates and low-quality text;
- records document-level provenance and deterministic train/validation assignment;
- trains a tokenizer on training text only;
- starts a fresh foundation-pretraining run with mixed precision and gradient accumulation;
- evaluates dialogue health and exports checkpoint/provenance artifacts;
- optionally continues on a selected OSCR code subset.

The first run is a pipeline-scale experiment, not a frontier model training run. Increase the scale after measuring tokens/second, peak GPU memory, validation behavior, and data quality.
