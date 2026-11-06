# First Artemus Cognitive Kernel — Implementation Milestone

**Status:** Proposed first executable milestone  
**Depends on:** [Artemus Cognitive System Architecture](../architecture/artemus-cognitive-system.md)

## Objective

Build the smallest testable runtime that demonstrates a complete perception-to-action feedback loop. The first implementation must work with a deterministic mock reasoner; no trained language model is required to prove that the control system is correct.

## Deliverable

A single-process, deterministic cognitive kernel that can:
1. Accept a typed goal with explicit success conditions.
2. Receive observations from a mock environment.
3. Read and write scoped, provenance-bearing memory.
4. Maintain a small, inspectable world state.
5. request a bounded plan from a replaceable reasoner interface.
6. Validate proposed actions against a permission policy.
7. Execute actions through a mock tool/environment adapter.
8. Verify the resulting observations against success conditions.
9. Persist the episode and resume or replay it.

## Proposed package boundary

```text
src/artemus/
  contracts/       # goals, events, observations, plans, actions, results
  kernel/          # state machine, scheduler, control loop, cancellation
  memory/          # typed memory interface and SQLite implementation
  world/           # entity/event state, claims, evidence, uncertainty
  reasoning/       # reasoner protocol and deterministic test reasoner
  planning/        # task graph, dependencies, expected outcomes
  actions/         # tool contracts, permissions, mock environment
  verification/    # acceptance criteria and independent checks
  learning/        # initially candidate records only; no online weight updates
tests/artemus/
  test_kernel.py
  test_memory.py
  test_world.py
  test_permissions.py
  test_recovery.py
  test_replay.py
```

This is a proposed boundary, not a requirement to create empty modules all at once. Add a package only when its contract is being implemented.

## Build sequence

### K0 — Contracts and deterministic control
- Define immutable, serialisable types for Goal, Task, Event, Observation, Plan, ActionProposal, ActionResult, and VerificationResult.
- Define explicit state transitions and failure states.
- Define protocols for Reasoner, MemoryStore, WorldModel, Planner, Environment, and Verifier.
- Implement a deterministic mock environment and mock reasoner.
- Test the full loop without any neural model.

**Exit:** the same input event sequence and configuration produce the same task trace and outcome.

### K1 — Durable episodic memory
- Store immutable events and task episodes in SQLite.
- Attach source, timestamp, scope, confidence/epistemic status, and content hash.
- Add query by task, entity, time, and provenance.
- Support deletion/retention policies without silently mutating historical audit records.
- Add crash/restart and replay tests.

**Exit:** a task survives process restart and resumes from the last verified checkpoint without duplicating completed external effects.

### K2 — Structured world state
- Represent entities, relations, events, properties, and constraints as versioned claims.
- Link every claim to one or more observations.
- Distinguish observed, inferred, predicted, disputed, and retracted information.
- Compare predicted outcomes with later observations.
- Test contradictions, stale claims, and source revocation.

**Exit:** the system can explain which evidence supports a current belief and revise it without rewriting the original episode.

### K3 — Planning and independent verification
- Add a task graph with dependencies, preconditions, budgets, risk, and acceptance criteria.
- Require proposed actions to name their expected observations.
- Verify results through environment responses and deterministic checks.
- Bound retries and re-plan after contradictory observations.

**Exit:** success, failure, timeout, cancellation, and recovery are all explicit and test-covered.

### K4 — Model adapters and multimodal expansion
- Connect ArtemisLM behind a versioned Reasoner interface.
- Keep model-generated proposals untrusted until validated.
- Add one input modality at a time through adapters.
- Compare the model adapter with the deterministic baseline on the same tasks.

**Exit:** changing the neural model does not require rewriting the kernel, memory, world model, or tool contracts.

### K5 — Candidate learning and promotion
- Convert reviewed episodes into candidate skills or training examples.
- Require independent validation and provenance for every candidate.
- Build a protected regression suite and evaluate candidate updates offline.
- Promote versioned artifacts only after review; preserve rollback.

**Exit:** improvements generalise to held-out tasks without unacceptable regressions or permission-boundary failures.

## Required quality gates

- Deterministic replay of a fixed scenario.
- No action outside the declared permission scope.
- No duplicate execution after retry or restart for idempotent operations.
- No success state without verifier evidence.
- No semantic memory claim without provenance.
- Conflicting evidence remains inspectable.
- Cancellation and timeouts do not leave the task marked complete.
- Model adapter failures are contained and reported.
- Persistent state can be inspected, exported, and deleted according to policy.
- All test cases run in a disposable environment with bounded resources.

## Explicitly deferred

Do not add a multi-agent swarm, distributed orchestration, vector database, learned world model, online reinforcement learning, automatic code self-modification, or production UI in K0–K2. Reconsider each only after a concrete bottleneck and a measurable test justify it.

## Relationship to the current model code

The existing `src/artemis` package remains the experimental neural language-model component. It should be wrapped by an adapter only after the kernel has stable contracts. Preserve existing checkpoints and command names during any later package migration; do not mix an AGI-system rename with an unverified model or checkpoint-format change.
