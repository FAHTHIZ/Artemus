"""Deterministic cognitive kernel and simulated environment for K0."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from .contracts import (
    ActionProposal,
    ActionResult,
    Goal,
    Observation,
    Plan,
    RunResult,
    RunStatus,
    VerificationResult,
)
from .memory import EpisodicMemory


class Reasoner(Protocol):
    def plan(self, goal: Goal, observation: Observation) -> Plan:
        """Propose the next actions needed to satisfy a goal."""


class Environment(Protocol):
    def observe(self) -> Observation:
        """Return the current externally observable state."""

    def execute(self, action: ActionProposal) -> ActionResult:
        """Execute one action within the environment's own constraints."""


class Verifier(Protocol):
    def verify(self, goal: Goal, observation: Observation) -> VerificationResult:
        """Check goal completion against observed state, independently of the reasoner."""


class GoalVerifier:
    """Verify exact key/value conditions against the latest observation."""

    def verify(self, goal: Goal, observation: Observation) -> VerificationResult:
        evidence: dict[str, Any] = {}
        passed = True
        for key, expected in goal.success_condition.items():
            actual = observation.state.get(key)
            matches = key in observation.state and actual == expected
            evidence[key] = {"expected": expected, "actual": actual, "matches": matches}
            passed = passed and matches
        reason = "all success conditions matched" if passed else "one or more conditions remain unmet"
        return VerificationResult(passed=passed, evidence=evidence, reason=reason)


class DeterministicReasoner:
    """Test reasoner that proposes one state update for the first unmet condition."""

    def plan(self, goal: Goal, observation: Observation) -> Plan:
        for key, expected in goal.success_condition.items():
            if key not in observation.state or observation.state[key] != expected:
                action = ActionProposal(
                    name="set_value",
                    arguments={"key": key, "value": expected},
                    required_permission="workspace_write",
                    rationale=f"Set {key!r} to the goal's expected value.",
                )
                return Plan(actions=(action,), rationale="Satisfy the first unmet goal condition.")
        return Plan(actions=(), rationale="The observation already satisfies the goal.")


class SimulatedEnvironment:
    """Small deterministic key/value world used to prove the control loop."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._state = deepcopy(initial_state or {})

    def observe(self) -> Observation:
        return Observation(state=deepcopy(self._state), source="simulated_environment")

    def execute(self, action: ActionProposal) -> ActionResult:
        if action.name != "set_value":
            return ActionResult(success=False, error=f"unsupported action: {action.name}")
        key = action.arguments.get("key")
        if not isinstance(key, str) or not key.strip():
            return ActionResult(success=False, error="set_value requires a non-empty string key")
        value = deepcopy(action.arguments.get("value"))
        previous = deepcopy(self._state.get(key))
        self._state[key] = value
        return ActionResult(
            success=True,
            output={"key": key, "previous": previous, "value": value},
        )


class CognitiveKernel:
    """Coordinate goal checks, planning, policy, action, and verification."""

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        environment: Environment,
        verifier: Verifier,
        memory: EpisodicMemory,
        allowed_permissions: set[str] | frozenset[str],
    ) -> None:
        self.reasoner = reasoner
        self.environment = environment
        self.verifier = verifier
        self.memory = memory
        self.allowed_permissions = frozenset(allowed_permissions)

    def run(self, goal: Goal) -> RunResult:
        task_id = goal.goal_id
        self.memory.record(
            task_id,
            "goal_created",
            {
                "description": goal.description,
                "success_condition": dict(goal.success_condition),
                "max_steps": goal.max_steps,
            },
        )

        steps = 0
        observation = self.environment.observe()
        verification = self.verifier.verify(goal, observation)

        for step in range(1, goal.max_steps + 1):
            self.memory.record(
                task_id,
                "observation",
                {"step": steps, "source": observation.source, "state": dict(observation.state)},
            )
            self.memory.record(
                task_id,
                "verification",
                {"step": steps, "passed": verification.passed, "evidence": dict(verification.evidence)},
            )
            if verification.passed:
                return self._finish(
                    goal,
                    RunStatus.COMPLETED,
                    steps,
                    observation,
                    verification,
                    "goal conditions verified",
                )

            plan = self.reasoner.plan(goal, observation)
            self.memory.record(
                task_id,
                "plan_proposed",
                {
                    "step": step,
                    "rationale": plan.rationale,
                    "actions": [action.name for action in plan.actions],
                },
            )
            if not plan.actions:
                return self._finish(
                    goal,
                    RunStatus.BLOCKED,
                    steps,
                    observation,
                    verification,
                    plan.rationale or "reasoner returned no actionable plan",
                )

            action = plan.actions[0]
            steps = step
            if action.required_permission not in self.allowed_permissions:
                self.memory.record(
                    task_id,
                    "action_denied",
                    {
                        "step": step,
                        "action": action.name,
                        "required_permission": action.required_permission,
                    },
                )
                return self._finish(
                    goal,
                    RunStatus.BLOCKED,
                    steps,
                    observation,
                    verification,
                    f"permission denied: {action.required_permission}",
                )

            result = self.environment.execute(action)
            self.memory.record(
                task_id,
                "action_result",
                {
                    "step": step,
                    "action": action.name,
                    "success": result.success,
                    "output": dict(result.output),
                    "error": result.error,
                },
            )
            if not result.success:
                return self._finish(
                    goal,
                    RunStatus.FAILED,
                    steps,
                    observation,
                    verification,
                    result.error or "environment rejected the action",
                )

            observation = self.environment.observe()
            verification = self.verifier.verify(goal, observation)

        self.memory.record(
            task_id,
            "final_observation",
            {"source": observation.source, "state": dict(observation.state)},
        )
        status = RunStatus.COMPLETED if verification.passed else RunStatus.LIMIT_REACHED
        reason = "goal conditions verified" if verification.passed else "step budget exhausted"
        return self._finish(goal, status, steps, observation, verification, reason)

    def _finish(
        self,
        goal: Goal,
        status: RunStatus,
        steps: int,
        observation: Observation,
        verification: VerificationResult,
        reason: str,
    ) -> RunResult:
        self.memory.record(
            goal.goal_id,
            "task_finished",
            {
                "status": status.value,
                "steps": steps,
                "verified": verification.passed,
                "reason": reason,
                "evidence": dict(verification.evidence),
            },
        )
        return RunResult(
            task_id=goal.goal_id,
            status=status,
            steps=steps,
            final_observation=observation,
            verification=verification,
            reason=reason,
        )
