"""Typed contracts for the first Artemus cognitive-kernel slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class RunStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class Goal:
    description: str
    success_condition: Mapping[str, Any]
    max_steps: int = 4
    goal_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("goal description must not be empty")
        if not self.success_condition:
            raise ValueError("goal requires at least one success condition")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.goal_id.strip():
            raise ValueError("goal_id must not be empty")


@dataclass(frozen=True, slots=True)
class Observation:
    state: Mapping[str, Any]
    source: str = "environment"

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("observation source must not be empty")


@dataclass(frozen=True, slots=True)
class ActionProposal:
    name: str
    arguments: Mapping[str, Any]
    required_permission: str = "read_only"
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("action name must not be empty")
        if not self.required_permission.strip():
            raise ValueError("required_permission must not be empty")


@dataclass(frozen=True, slots=True)
class Plan:
    actions: tuple[ActionProposal, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            object.__setattr__(self, "actions", tuple(self.actions))


@dataclass(frozen=True, slots=True)
class ActionResult:
    success: bool
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    evidence: Mapping[str, Any]
    reason: str


@dataclass(frozen=True, slots=True)
class RunResult:
    task_id: str
    status: RunStatus
    steps: int
    final_observation: Observation
    verification: VerificationResult
    reason: str
