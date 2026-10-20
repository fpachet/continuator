"""
Small constraint-building API for finite sequence generation.

Constraints compile to the external vo_regular_bp ConstraintSet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ConstraintProblem:
    length: int | None = None
    allowed_values_by_position: dict[int, set[Any]] = field(default_factory=dict)

    def at(self, position: int) -> "PositionConstraint":
        return PositionConstraint(self, position)

    def require(self, position: int, value: Any) -> "ConstraintProblem":
        self.allowed_values_by_position[position] = {value}
        return self

    def require_one_of(self, position: int, values: Iterable[Any]) -> "ConstraintProblem":
        allowed_values = set(values)
        if not allowed_values:
            raise ValueError("A positional constraint cannot allow an empty set.")
        self.allowed_values_by_position[position] = allowed_values
        return self



@dataclass(frozen=True)
class PositionConstraint:
    problem: ConstraintProblem
    position: int

    def equals(self, value: Any) -> ConstraintProblem:
        return self.problem.require(self.position, value)

    def one_of(self, values: Iterable[Any]) -> ConstraintProblem:
        return self.problem.require_one_of(self.position, values)
