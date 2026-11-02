# OASIS — Agent Runtime Architecture

**Status:** Initial design proposal.

## 1. Purpose

OASIS is the execution environment around Artemis. It turns a user goal into a controlled, observable, verifiable workflow. OASIS is not the foundation model and should remain useful as model checkpoints evolve.

## 2. Core components

### Orchestrator
Owns task lifecycle, cancellation, retries, deadlines, and state transitions.

### Planner
Converts a request into goals, sub-tasks, dependencies, acceptance criteria, and a completion definition. Plans are revisable when new observations invalidate assumptions.

### Context engine
Collects only relevant information from conversation, project files, memory, indexes, and tool observations. It tracks provenance and relevance.

### Model adapter
Calls a versioned Artemis inference API. It handles timeouts, token budgets, streaming, structured outputs, and model-version selection.

### Tool manager
Exposes typed tools with schemas, permissions, timeouts, and result validation. Model output is a proposal, not authorisation.

### Execution environments
Run code and commands in isolated workspaces. Support resource limits, network policy, dependency caching, and cleanup.

### Verification engine
Runs tests and checks selected for the task: unit/integration tests, type checks, linters, builds, static analysis, schema validation, or independent review.

### Session and memory store
Persists task state, intermediate artifacts, user-approved preferences, project facts, and resumable checkpoints under explicit retention and deletion rules.

### Audit and observability
Records task IDs, model/checkpoint version, tool name, arguments or safe summaries, permission decisions, result codes, timings, and verification artifacts. Secrets must be redacted.

## 3. Task lifecycle

```text
CREATED
  -> UNDERSTANDING
  -> PLANNING
  -> CONTEXT_ASSEMBLY
  -> EXECUTION
  -> VERIFICATION
      -> EXECUTION (bounded retry after failure)
      -> NEEDS_USER_INPUT
      -> BLOCKED
      -> COMPLETED
  -> ARCHIVED
```

Every transition should be explicit. Retries require a reason and have configurable limits. The agent must be able to stop, pause, resume, and report incomplete work.

## 4. Tool contract

Every tool definition should include:
- Stable identifier and version.
- Human-readable purpose.
- JSON-compatible input schema.
- Output schema and error cases.
- Required permission level.
- Whether the action is read-only, writes local state, or affects external state.
- Timeout and resource limits.
- Audit and redaction rules.

Tool execution must validate arguments independently of the model. A tool response is untrusted input and must not be treated as an instruction to bypass the runtime's rules.

## 5. Permission tiers

Suggested initial tiers:

1. **Read-only:** inspect approved project files and search indexes.
2. **Workspace write:** create or modify files inside a designated project workspace.
3. **Sandbox execution:** run commands and tests in an isolated environment.
4. **External side effects:** publish, deploy, send messages, spend money, alter accounts, or mutate remote systems; require explicit permission and, where appropriate, per-action confirmation.

Never expose unrestricted host access, private credentials, or unrestricted network access to a model-generated command. Use least privilege, allowlists, secrets isolation, and human approval for consequential actions.

## 6. Coding workflow

For a repository task:
1. Detect project type, language, package manager, and key entry points.
2. Inspect repository status and preserve pre-existing user changes.
3. Index symbols, imports, call references, tests, and configuration.
4. Write acceptance criteria and a task plan.
5. Retrieve a focused context slice.
6. Propose a minimal patch.
7. Apply patch in an isolated branch/worktree or sandbox.
8. Run targeted tests, then appropriate broader checks.
9. Inspect the final diff for unrelated changes and secrets.
10. Return changed files, test commands/results, and remaining risks.

If a test cannot run, report that fact. Never report a test as passing without the corresponding result.

## 7. Multi-agent execution

Begin with one Artemis instance and a deterministic orchestrator. Add parallel workers only after the single-agent workflow is reliable.

Potential worker roles:
- Research / repository explorer.
- Architect / plan reviewer.
- Implementer.
- Test engineer.
- Independent code reviewer.

Workers should have separate task scopes and, where practical, independent contexts. The orchestrator resolves disagreements based on evidence and acceptance criteria rather than majority vote.

## 8. Memory policy

Memory writes should be typed and scoped. Distinguish transient task state from durable user or project memory. Attach provenance, confidence, timestamps, and deletion controls. Do not convert every model-generated claim into a trusted fact.

## 9. Reliability requirements

- Bounded retries and deadlines.
- Idempotency keys for retriable external operations.
- Durable checkpoints for long-running work.
- Cancellation and recovery.
- Clear tool failure taxonomy.
- Structured logs with secret redaction.
- Independent verification of consequential output.
- Human approval for destructive or irreversible actions.

## 10. Initial implementation order

1. CLI/API request interface.
2. Single-model adapter (initially a mock adapter until Artemis inference exists).
3. Typed tool registry.
4. Sandboxed workspace and command runner.
5. Task-state persistence.
6. Verification hooks.
7. Basic retrieval and project index.
8. Multi-agent orchestration only after baseline reliability is measured.
