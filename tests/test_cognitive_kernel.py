from pathlib import Path

from artemus.contracts import Goal, Plan, RunStatus
from artemus.kernel import (
    CognitiveKernel,
    DeterministicReasoner,
    GoalVerifier,
    SimulatedEnvironment,
)
from artemus.memory import EpisodicMemory


def make_kernel(
    memory: EpisodicMemory,
    environment: SimulatedEnvironment,
    *,
    permissions: set[str] | None = None,
) -> CognitiveKernel:
    return CognitiveKernel(
        reasoner=DeterministicReasoner(),
        environment=environment,
        verifier=GoalVerifier(),
        memory=memory,
        allowed_permissions=permissions or {"read_only", "workspace_write"},
    )


def test_kernel_completes_goal_only_after_observed_verification(tmp_path: Path) -> None:
    database = tmp_path / "episodes.sqlite"
    goal = Goal(
        description="Prepare the simulated workspace",
        success_condition={"workspace": "ready"},
    )
    environment = SimulatedEnvironment({"workspace": "empty"})

    with EpisodicMemory(database) as memory:
        result = make_kernel(memory, environment).run(goal)
        event_types = [event.event_type for event in memory.events(goal.goal_id)]

    assert result.status is RunStatus.COMPLETED
    assert result.steps == 1
    assert result.verification.passed is True
    assert result.final_observation.state["workspace"] == "ready"
    assert "action_result" in event_types
    assert "task_finished" == event_types[-1]

    # The event history is durable across closing and reopening the database.
    with EpisodicMemory(database) as reopened:
        persisted = reopened.events(goal.goal_id)
    assert len(persisted) == len(event_types)
    assert persisted[-1].payload["verified"] is True


def test_kernel_does_not_claim_success_without_evidence() -> None:
    goal = Goal(description="Reach ready state", success_condition={"state": "ready"})
    environment = SimulatedEnvironment({"state": "waiting"})
    with EpisodicMemory() as memory:
        result = make_kernel(memory, environment, permissions={"read_only"}).run(goal)

    assert result.status is RunStatus.BLOCKED
    assert result.verification.passed is False
    assert environment.observe().state["state"] == "waiting"
    assert "permission denied" in result.reason


def test_kernel_completes_without_action_when_goal_is_already_true() -> None:
    goal = Goal(description="Check readiness", success_condition={"state": "ready"})
    environment = SimulatedEnvironment({"state": "ready"})
    with EpisodicMemory() as memory:
        result = make_kernel(memory, environment).run(goal)

    assert result.status is RunStatus.COMPLETED
    assert result.steps == 0
    assert result.verification.evidence["state"]["matches"] is True


class NoPlanReasoner:
    def plan(self, goal: Goal, observation) -> Plan:
        return Plan(actions=(), rationale="Need more information.")


def test_kernel_reports_blocked_when_reasoner_has_no_plan() -> None:
    goal = Goal(description="Do something", success_condition={"x": 1})
    with EpisodicMemory() as memory:
        kernel = CognitiveKernel(
            reasoner=NoPlanReasoner(),
            environment=SimulatedEnvironment(),
            verifier=GoalVerifier(),
            memory=memory,
            allowed_permissions={"workspace_write"},
        )
        result = kernel.run(goal)

    assert result.status is RunStatus.BLOCKED
    assert result.steps == 0
    assert result.reason == "Need more information."
