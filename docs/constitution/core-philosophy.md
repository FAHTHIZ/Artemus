# Artemis Core Philosophy

**Status:** Foundational project invariant  
**Applies to:** the ArtemisLM neural-model component, its data, architecture, training, evaluation, and inference. Whole-system principles for Artemus are defined in [Artemus Cognitive System Architecture](../architecture/artemus-cognitive-system.md).

## 1. ArtemisLM is our independently trained model component

Artemis is an independently designed and independently trained foundation model. Every official foundation checkpoint must have a traceable lineage that begins with random initialization inside this project.

Not permitted:
- Importing another model's trained weights and renaming or repackaging them as Artemis.
- Hiding a pretrained model inside the claimed foundation checkpoint.
- Claiming another model's answer, benchmark result, or capability as an Artemis result.

We can read papers, reproduce public research, implement published ideas, use appropriately licensed code and data, and compare against existing models. Learning from scientific literature is not copying learned weights. Any external code, dataset, or reference must have recorded origin, revision, licence, and use conditions.

External models may be used as independent evaluators after the relevant benchmark and privacy rules are checked. They are not our hidden base model and are not default sources of training answers.

## 2. Checkpoint provenance is mandatory

Each released checkpoint must record:
- Root random-initialization seed and source-code commit.
- Full parent-checkpoint SHA-256 chain.
- Model architecture, tokenizer model/hash, and parameter count.
- Dataset manifest, source revisions, licences, split definitions, preprocessing version, and hashes.
- Optimizer, schedule, token/step counts, compute device, wall time, and run configuration.
- Evaluation suite versions, per-task outcomes, limitations, and artifact hashes.

A checkpoint with missing or ambiguous ancestry cannot be labelled an official Artemis release. When a historical checkpoint lacks provenance, report that honestly; never retroactively invent its origin.

## 3. Capability per resource—not a parameter-count contest

Our design goal is the best achievable capability per:
- Total stored parameter.
- Active parameter per token.
- Training FLOP and training token.
- Inference FLOP, memory byte, latency, and energy.
- Unit of actual compute cost.

Always distinguish **total parameters** from **active parameters**. A sparse model with 40B total parameters and 4B active per token is not a 4B-parameter model. Also report expert storage, KV cache, activations, optimizer state, and runtime overhead.

The target is not “the smallest model at any cost.” It is the best Pareto point that preserves the breadth and reliability of a flagship general model.

## 4. Frontier flagship ambition

Artemis's long-term ambition is a compact, independently trained flagship capable of frontier-level software engineering, reasoning, mathematics, multilingual communication, knowledge work, and reliable tool use.

This is a target, not a promise. We will not claim parity with GPT-6 Astra, GPT-6.1 Sol, Claude Fable 5.1, or any other reference from parameter counts, training loss, a single benchmark, or a compelling demo. Parity must be supported by comparable independent evaluations across the intended capability areas.

A strong coding model is an important milestone, not proof of general frontier intelligence.

## 5. Candid, intellectually open behaviour

Artemis should be direct, intellectually open, and minimally arbitrary in what it discusses. It should not refuse ordinary questions merely because a subject is controversial, uncomfortable, political, religious, sexual in a non-exploitative context, or critical of an institution. It should distinguish discussion, analysis, fiction, and education from taking action in the world.

“Uncensored” in this project means candid inquiry and no arbitrary viewpoint-based suppression. It does not mean discarding privacy, consent, lawful access controls, or proportionate safeguards against enabling serious real-world harm. When a concrete boundary applies, the model should explain it plainly and remain helpful within that boundary.

It must say when it is uncertain, avoid fabricating sources or actions, and never claim that a tool succeeded unless the tool actually reported success.

## 6. Data integrity and licensing

Publicly reachable data is not automatically free to use. Only ingest data with an appropriate documented permission or licence basis for the intended training and distribution.

Default rules:
- No private user conversations without explicit consent and a defined purpose.
- No credentials, secrets, or unnecessary personal data.
- No bypassing paywalls, access controls, repository restrictions, or dataset opt-outs.
- No unlicensed books, code, private repositories, or proprietary corpora.
- No external-model-generated response corpus by default.
- Preserve original licence, attribution, source revision, hashes, and removal obligations.
- Quarantine sources whose rights are unclear.
- Do not apply viewpoint, political, religious, or ideological filters to lawful material merely because of its position. Data-quality filters must target documented issues such as duplication, corruption, spam, privacy/PII, provenance gaps, access restrictions, or concrete serious-harm risks—not disagreement or discomfort.

## 7. Learning from conversations without uncontrolled self-modification

Conversation feedback can create candidate lessons or test cases. It must not silently rewrite production weights. Any model update requires curated data, source lineage, a reproducible run, independent evaluations, regression tests, safety tests, and a rollback checkpoint.

## 8. Evidence before promotion

Preserve failed runs and negative results. Compare under pinned benchmarks, multiple seeds where practical, and the same inference/tool budget. A candidate is better only when it improves the desired capability without unacceptable regressions. Be ambitious about goals and unsentimental about evidence.
