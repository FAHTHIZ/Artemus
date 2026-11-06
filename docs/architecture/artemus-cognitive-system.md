# Artemus — Cognitive System Architecture

**Status:** Foundational architecture proposal  
**Scope:** The complete Artemus system, not only its neural language model  
**Evidence rule:** This document defines a research and engineering direction. It does not claim that Artemus currently has AGI-level capabilities.

## 1. Identity and system boundary

**Artemus is the complete cognitive system. It is not synonymous with an LLM.** A language model is one possible reasoning and communication component inside a larger system that must perceive, maintain state, learn from experience, plan, act, and evaluate outcomes.

The repository currently contains a small, independently trained decoder-only model under `src/artemis`, plus training, checkpoint, data-preparation, and evaluation utilities. It also contains proposed OASIS runtime documentation. It does **not** yet contain an implemented general cognitive runtime, persistent cognitive memory, a predictive world model, multimodal perception, a general planner, or a verified lifelong-learning loop.

Keep these concepts distinct:

- **Artemus:** the whole cognitive system and research programme.
- **ArtemisLM:** the current experimental neural language-model component. Its existing package and checkpoint names remain compatible while a deliberate migration is designed.
- **Cognitive kernel:** the control system that coordinates goals, state, memory, reasoning, actions, and verification.
- **OASIS:** a separately developed execution/runtime capability. Artemus may connect to it through explicit interfaces; it must not depend on OASIS being the source of all intelligence.
- **Training and evaluation platform:** the reproducible pipeline for improving and testing learned components. It is not the cognitive system itself.

A model checkpoint, a collection of tools, a memory database, or a multi-agent setup alone is not Artemus.

## 2. Design principles

1. **System-level intelligence:** evaluate the integrated perception–memory–reasoning–planning–action loop, not only text generation.
2. **Model independence:** the cognitive kernel must work with a deterministic mock reasoner and support interchangeable learned models and specialist engines.
3. **Predictive state:** maintain an explicit, revisable model of relevant entities, relationships, events, constraints, and expected outcomes.
4. **Memory with a lifecycle:** preserve experience, retrieve it selectively, consolidate repeated evidence, track contradictions, and support correction and deletion.
5. **Grounded action:** model-generated plans are proposals. A policy gate authorises actions, tools return observations, and independent checks verify outcomes.
6. **Learning without uncontrolled self-modification:** online experience may update scoped memory and candidate skills; production model weights and runtime code change only through reviewed, reproducible promotion.
7. **One coherent controller before many agents:** establish reliable single-system control before adding parallel workers or distributed orchestration.
8. **Evidence over labels:** use measured task generalisation, reliability, and resource efficiency. Do not call the system AGI because it has a cognitive architecture.

## 3. Cognitive architecture

```text
                HUMAN / ENVIRONMENT / DIGITAL SYSTEMS
                              |
                    INPUT & PERCEPTION
             text · image · audio · video · events
                              |
                  STATE ESTIMATION / WORLD MODEL
           entities · relations · time · constraints
                 predictions · uncertainty
                              |
                    COGNITIVE KERNEL
          goals · attention · scheduling · budgets
           interrupts · state transitions · control
                 /          |           \
                /           |            \
          MEMORY         REASONING       SELF-MODEL
       working             neural        capabilities
       episodic            symbolic      uncertainty
       semantic            retrieval     resource use
       procedural          simulation    error history
       sensory             specialist
                \           |            /
                 \          |           /
                   PLANNING & DECISION
            goal graph · dependencies · alternatives
              expected effects · risk · contingencies
                              |
                      POLICY / PERMISSION GATE
                              |
                    ACTION & ENVIRONMENT I/O
               tools · APIs · code · devices · services
                              |
                     OBSERVATIONS / RESULTS
                              |
                 VERIFICATION & CREDIT ASSIGNMENT
                              |
               MEMORY UPDATE / MODEL REVISION / REPLAN
                              |
                     CONTROL LOOP REPEATS

       Separate offline lane:
       curated data -> candidate learning -> evaluation
                    -> approval -> versioned promotion
```

### 3.1 Input and perception

Represent inputs as typed events with source, timestamp, modality, session, access scope, and integrity metadata. Perception adapters transform raw input into candidate observations. They must preserve links to the original source and distinguish direct observations from inferred interpretations.

Start with text and structured tool observations. Add image, audio, video, and other sensors through replaceable adapters; do not make the first runtime depend on a large multimodal model.

### 3.2 World model and state estimation

The world model is not just a vector-search index or a transcript summary. It is a structured, time-aware representation of what the system currently believes about the task and environment.

A record should be able to express:
- Entity, relationship, event, property, and constraint.
- Source observation and provenance.
- Valid time and recorded time.
- Confidence or uncertainty, plus the method used to derive it.
- Whether the record is observed, inferred, predicted, disputed, superseded, or retracted.
- Links to supporting and contradicting evidence.

Predictions must be compared with later observations. Incorrect beliefs must be revisable without rewriting the historical event that caused them.

### 3.3 Memory

Use distinct memory functions rather than one undifferentiated chat-history table:

| Memory | Purpose | Initial implementation |
|---|---|---|
| Working | Active task state, current context, intermediate results | Bounded in-memory state |
| Episodic | What happened, when, in which environment, and with what result | Append-only SQLite event records |
| Semantic | Reusable facts and relationships with evidence | Versioned claims linked to episodes |
| Procedural | Validated skills, methods, and executable workflows | Versioned skill records with tests |
| Sensory | References to source observations and derived features | Content-addressed metadata and local references |
| Self-model | Known capabilities, limitations, calibration, and system configuration | Versioned capability and evaluation records |

Memory writes require a type, scope, provenance, and retention policy. A generated statement is not automatically a trusted fact. Consolidation should merge genuinely repeated information while preserving source links; contradictions must remain visible. Forgetting may reduce retrieval priority, but must not silently erase audit records or violate deletion requests.

### 3.4 Reasoning substrate

The system should select the reasoning method that fits the problem. Candidate components include:

- Neural language models for language understanding, synthesis, and flexible proposal generation.
- Symbolic and constraint solvers for exact relations, invariants, and structured problems.
- Calculators and domain-specific numerical libraries for arithmetic and scientific computation.
- Retrieval engines for source-grounded information.
- Sandboxed code execution for executable hypotheses and tests.
- Simulators or learned predictive models for comparing possible actions and their expected effects.

These are complementary tools, not a committee of chatbots whose votes are treated as truth. The cognitive kernel owns arbitration, evidence requirements, and resource allocation.

### 3.5 Goal management and planning

Represent goals and plans as explicit graphs. Each task should include:
- Desired outcome and measurable completion criteria.
- Preconditions, dependencies, and relevant state.
- Candidate actions and expected observations.
- Resource, time, and context budgets.
- Risks, reversibility, and required permissions.
- Fallbacks and conditions that trigger replanning.

Use hierarchical decomposition where it helps, but do not assume that a longer plan is a better plan. Execute a bounded portion, inspect the real outcome, update state, and re-plan when assumptions fail.

### 3.6 Action and environment interface

Every action has a typed input/output contract, a permission level, timeout, resource limits, and an audit record. Separate read-only observations from workspace changes and external side effects. Use least privilege, isolated execution, and explicit approval for consequential external actions.

An action is complete only when its result is returned and checked. Text that says an action happened is not evidence that it happened.

### 3.7 Self-model and metacognition

Maintain an inspectable description of what components are available, what they have actually demonstrated, their known failure modes, current resource budgets, and uncertainty calibration. The system should be able to defer, request missing information, choose a different method, or stop when it cannot justify the next action.

Self-reflection is useful only when it changes a measurable decision or improves error detection. Reflection text by itself is not learning.

### 3.8 Learning and adaptation

Separate three update timescales:

1. **Within-task adaptation:** update working state, retrieve relevant memories, revise a plan, and change strategy based on observed outcomes.
2. **Cross-task learning:** extract candidate facts, skills, and failure cases. Require provenance, validation, versioning, and regression checks before they become reusable.
3. **Model learning:** build curated datasets and train a candidate checkpoint offline. Preserve full data and checkpoint lineage, evaluate against protected holdouts, and promote or roll back deliberately.

Never allow a conversation or a model-generated reflection to silently rewrite production weights, executable skills, permissions, or the cognitive kernel.

## 4. The core cognitive cycle

The first executable system should implement this loop with deterministic components:

1. Receive an input event and validate its source and scope.
2. Convert it into candidate observations while preserving provenance.
3. Update the current world state and mark unresolved uncertainty.
4. Retrieve only memories relevant to the active goal.
5. Assess goals, constraints, deadlines, and available resources.
6. Generate candidate explanations, plans, or actions using the selected reasoning components.
7. Check preconditions and permissions before dispatching an action.
8. Execute through a bounded environment adapter.
9. Compare the observed result with the predicted result and the task's acceptance criteria.
10. Record the episode, update supported beliefs, and trigger a bounded re-plan or completion.

The control loop must support pause, cancellation, interruption, durable checkpoints, replay, and recovery after failure.

## 5. First implementation boundary

Do not begin by implementing every modality, training a much larger model, or creating a distributed swarm of agents. The first milestone is a deterministic cognitive kernel that can complete a small, observable task in a simulated environment using a mock reasoner.

The first vertical slice should prove:
- A goal can be represented with explicit acceptance criteria.
- The kernel can retrieve scoped memory and inspect world state.
- A plan can be converted into a permission-checked action.
- The environment returns an observation.
- The verifier can detect success and failure independently.
- The resulting episode can be persisted and replayed.
- A failed step can be resumed without inventing success.

Only after this works should ArtemisLM be connected as one reasoning adapter. The model's raw capability and the integrated system's performance must remain separately measurable.

## 6. Non-goals for the first milestone

- No claim of AGI or autonomous self-improvement.
- No automatic production-weight updates from live interactions.
- No unrestricted host, filesystem, network, credential, or device access.
- No vector database, distributed event bus, multi-agent swarm, or cloud microservice fleet without a measured need.
- No giant-model training before the data pipeline, hardware budget, and evaluation protocol justify it.
- No assumption that a language model's internal representations already provide a reliable world model or durable memory.

## 7. Research grounding

These works inform hypotheses to test; they are not turnkey designs or proof that Artemus will achieve general intelligence:

- Richens, Everitt, and Abel, **General Agents Need World Models** (ICML 2025): https://proceedings.mlr.press/v267/richens25a.html
- Huang et al., **Rethinking Memory Mechanisms of Foundation Agents in the Second Half** (2026 survey): https://arxiv.org/abs/2602.06052
- Kerestecioglu et al., **Human-Inspired Memory Architecture for LLM Agents** (May 2026): https://www.microsoft.com/en-us/research/publication/human-inspired-memory-architecture-for-llm-agents/
- Yang et al., **Scaling Automatic Research Agents via World Models** (August 2026 preprint): https://arxiv.org/abs/2608.12564
- Hakim et al., **Neuro-symbolic Agentic AI: Architectures, Integration Patterns, Applications, Open Challenges and Future Research Directions** (2026 survey): https://doi.org/10.1016/j.cosrev.2026.100902

## 8. Decision rule

Prefer the smallest, clearest implementation that can be independently tested. Every new subsystem must state:
- Which observed failure it addresses.
- Its input and output contracts.
- What evidence it stores and how it handles uncertainty.
- How it can fail and how the kernel recovers.
- How its contribution will be measured against a simpler baseline.

The architecture is successful only when the integrated system demonstrates capabilities that its individual components could not reliably deliver alone.
