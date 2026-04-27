"""
Small constraint-building API for finite sequence generation.

The first target is positional hard constraints. They compile to unary masks for
the chain solver, while still being convertible to the legacy dict format when
all constraints are single-value equality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


RawConstraints = Mapping[int, Any]


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

    def shifted(self, offset: int) -> "ConstraintProblem":
        return ConstraintProblem(
            length=None if self.length is None else self.length + offset,
            allowed_values_by_position={
                position + offset: set(values)
                for position, values in self.allowed_values_by_position.items()
            },
        )

    def without_position(self, position_to_remove: int) -> "ConstraintProblem":
        return ConstraintProblem(
            length=self.length,
            allowed_values_by_position={
                position: set(values)
                for position, values in self.allowed_values_by_position.items()
                if position != position_to_remove
            },
        )

    def has_constraint_at(self, position: int) -> bool:
        return position in self.allowed_values_by_position

    def single_value_at(self, position: int) -> Any | None:
        values = self.allowed_values_by_position.get(position)
        if values is None or len(values) != 1:
            return None
        return next(iter(values))

    def to_legacy_constraints(self) -> dict[int, Any]:
        legacy = {}
        for position, values in self.allowed_values_by_position.items():
            if len(values) != 1:
                raise ValueError(
                    "Only single-value constraints can be converted to the legacy dict format."
                )
            legacy[position] = next(iter(values))
        return legacy

    def to_allowed_indices(self, vp2index: dict[Any, int]) -> dict[int, set[int]]:
        result = {}
        for position, values in self.allowed_values_by_position.items():
            result[position] = {vp2index[value] for value in values}
        return result


@dataclass(frozen=True)
class PositionConstraint:
    problem: ConstraintProblem
    position: int

    def equals(self, value: Any) -> ConstraintProblem:
        return self.problem.require(self.position, value)

    def one_of(self, values: Iterable[Any]) -> ConstraintProblem:
        return self.problem.require_one_of(self.position, values)


def shift_constraints(constraints: ConstraintProblem | RawConstraints, offset: int):
    if isinstance(constraints, ConstraintProblem):
        return constraints.shifted(offset)
    return {position + offset: value for position, value in constraints.items()}


def without_constraint_at(constraints: ConstraintProblem | RawConstraints, position_to_remove: int):
    if isinstance(constraints, ConstraintProblem):
        return constraints.without_position(position_to_remove)
    return {
        position: value
        for position, value in constraints.items()
        if position != position_to_remove
    }


def has_constraint_at(constraints: ConstraintProblem | RawConstraints, position: int) -> bool:
    if isinstance(constraints, ConstraintProblem):
        return constraints.has_constraint_at(position)
    return position in constraints


def single_value_at(constraints: ConstraintProblem | RawConstraints, position: int) -> Any | None:
    if isinstance(constraints, ConstraintProblem):
        return constraints.single_value_at(position)
    return constraints.get(position)
