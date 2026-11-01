# Artemis / OASIS — System Architecture

**Status:** Initial architecture baseline (v0.1)  
**Decision maturity:** Proposed. Implementation choices remain subject to hardware discovery and measured experiments.

## 1. Product objective

Build a general-purpose foundation model from random initialization, then surround it with a reliable agent system. Coding and software engineering are first-class capability targets, not a narrow plugin.

The project has two distinct names and responsibilities:

- **Artemis** is the learned model: parameters, tokenizer, inference, and model-serving contract.
- **OASIS** is the agent runtime: task planning, context assembly, tools, execution, persistent state, and verification.

The separation prevents a model upgrade from requiring a complete rewrite of the agent environment.

## 2. Top-level architecture

```text
                         USER / API / CLI
                                |
                       Interface & Session API
                                |
                     OASIS ORCHESTRATOR
             +------------------+------------------+
             |                  |                  |
          Planner          Context Engine      Policy Gate
             |                  |                  |
             +------------------+------------------+
                                |
                         ARTEMIS INFERENCE
                    +-----------+-----------+
                    |                       |
               Model Runtime           Tool-call output
                    |                       |
             Token / KV cache          Tool Manager
                                            |
                                  Sandboxed Execution
                           +----------------+----------------+
                           |                |                |
                        Filesystem       Terminal         External APIs
                           |                |                |
                           +----------------+----------------+
                                            |
                                      Observations
                                            |
                                  Verification Engine
                                            |
                             Result / retry / escalation
                                            |
                                  Session & Memory Store

  Separate supporting systems:
  - Training platform and dataset registry
  - Evaluation platform and regression suite
  - Model/checkpoint registry
  - Retrieval indexes and document stores
  - Observability, audit records, and access control
```

## 3. Architectural domains

### 3.1 Foundation model

Responsibilities:

- Tokenize and interpret text, source code, and structured inputs.
- Generate responses and structured tool-call proposals.
- Perform language understanding, coding, mathematics, reasoning, and planning.
- Support long-context inference within tested memory and latency limits.
- Expose a versioned inference interface independent of OASIS internals.

The initial research baseline should be a small decoder-only Transformer trained from random initialization. Dense and sparse/MoE variants are research candidates. Do not commit to a very large MoE design until controlled scaling experiments show that its routing, quality, and systems complexity are justified.

### 3.2 OASIS agent runtime

Responsibilities:

- Convert requests into explicit goals and dependencies.
- Assemble task-specific context from current conversation, project files, and memory.
- Select tools and validate tool arguments.
- Run work in isolated environments with scoped permissions.
- Capture observations and verify outcomes.
- Manage retries, checkpoints, cancellation, and recovery.
- Keep an auditable record of consequential actions.

OASIS must not treat model-generated text as proof that an action succeeded. Tool results and independent checks are the source of truth.

### 3.3 Knowledge and memory

Knowledge retrieval is external to model parameters. It may contain documents, source repositories, specifications, and user-approved reference material. Every indexed item should retain its source, timestamp or revision, access policy, and provenance.

Memory is divided into:
- **Working memory:** active goals, plans, observations, and intermediate state.
- **Episodic memory:** previous task/session events and outcomes.
- **Semantic memory:** validated facts and relationships.
- **Procedural memory:** reusable methods and workflows.
- **Project memory:** repository structure, design decisions, known issues, and progress.

Retrieval should combine lexical search, vector search, and structured indexes. For code, include symbol and dependency relationships (for example, definitions, references, imports, call sites, tests, and configuration). Retrieve small relevant slices first and expand based on evidence.

### 3.4 Training platform

Responsibilities:

- Dataset ingestion, validation, filtering, deduplication, and provenance.
- Tokenizer experiments and corpus statistics.
- Reproducible training configurations and distributed execution.
- Checkpointing, resuming, and artifact registration.
- Supervised and preference/post-training data management.
- Candidate model evaluation and release promotion.

No dataset enters training without documented source, licence/use basis, quality checks, and applicable privacy review.

### 3.5 Evaluation platform

Evaluation is independent from training data wherever possible. It measures general capability, code correctness, mathematics, tool use, reliability, long-horizon completion, latency, and compute efficiency. Tests must include hidden or newly authored tasks to reduce benchmark overfitting.

## 4. The data path for a coding task

1. The user requests a change.
2. OASIS parses the request and creates a plan with acceptance criteria.
3. The context engine indexes or queries the repository and retrieves relevant files, symbols, tests, and documentation.
4. Artemis proposes a patch or tool action.
5. The tool manager checks permissions and executes in a sandbox or explicitly approved workspace.
6. The verification engine runs targeted tests, type checks, linting, build checks, and relevant regression tests.
7. Failures become observations. OASIS returns the failure context to Artemis for diagnosis and another attempt, within configured limits.
8. A final reviewer checks scope, diffs, tests, and outstanding uncertainty.
9. OASIS reports what changed and what was actually verified.

## 5. Resource and deployment strategy

The architecture must support multiple deployment profiles:

- **Local development:** workstation hosts source code, data tooling, small experiments, local inference, and the OASIS control plane.
- **Single-node accelerator:** a practical intermediate training and inference environment.
- **Distributed training:** multi-accelerator jobs with explicit parallelism, sharded checkpoints, and fault recovery.
- **Remote inference:** model server can run separately from the OASIS control plane.

Hardware has not yet been inventoried. Therefore accelerator count, model dimensions, context length, expert count, batch size, precision, and distributed strategy are intentionally not fixed in this document.

## 6. Core interfaces

Define explicit versioned contracts before connecting components:

- **Inference API:** messages, tokenization/model version, generation settings, structured tool calls, usage, and errors.
- **Tool API:** tool name, typed arguments, required permissions, timeout, result schema, and audit metadata.
- **Memory API:** scoped query, provenance-bearing results, write/update policy, and deletion/retention semantics.
- **Training artifact API:** dataset/configuration/checkpoint identifiers, parent lineage, metrics, and code revision.
- **Evaluation API:** test-suite version, environment, model checkpoint, raw result, aggregate score, and failure artifacts.

Keep the first implementation minimal. Do not build a distributed microservice for every module before there is a working end-to-end baseline.

## 7. Non-goals for the bootstrap phase

- Claiming frontier-level capability before evaluation supports it.
- Choosing a trillion-parameter configuration before hardware and scaling experiments.
- Allowing every chat to modify model weights online.
- Giving an agent unrestricted access to a user's machine or credentials.
- Storing large datasets or model weights in Git.
- Treating benchmark scores as a substitute for verified real-project performance.

## 8. Open decisions

These decisions require evidence or user input:

1. Canonical product spelling and naming.
2. Workstation CPU, GPU/VRAM, RAM, storage, and operating system.
3. Initial development framework and supported accelerator backend.
4. Initial tokenizer design and vocabulary size.
5. Initial baseline model dimensions and context length.
6. Data sources, licensing policy, and language/code mix.
7. Local-only versus hybrid remote compute deployment.
8. OASIS tool permissions and approval boundaries.
9. Initial evaluation suite and release thresholds.

## 9. Architecture invariants

- Artemis is trained from random initialization.
- OASIS and Artemis have separate versioned interfaces.
- Training data and model artifacts have provenance and lineage.
- Memory retrieval is distinct from neural model weights.
- Actions are verified using tool outputs or independent checks.
- Model updates are evaluated, reversible, and promoted deliberately.
- Hardware-specific choices are measured rather than assumed.
