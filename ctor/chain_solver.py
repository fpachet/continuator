"""
Iterative sum-product inference for finite Markov chains.

This module computes the same chain marginals as the generic belief-propagation
graph used in variable_order_markov.py, but with an explicit forward-backward
schedule instead of recursive message calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NoSolutionErrorInChainSolver(Exception):
    """Raised when transition and unary constraints leave no feasible sequence."""


@dataclass(frozen=True)
class ForwardBackwardResult:
    marginals: np.ndarray
    forward: np.ndarray
    backward: np.ndarray


class SparseForwardBackward:
    """
    Forward-backward solver for a chain with transition matrix P[prev, next].

    Unary potentials are provided as a matrix of shape (length, vocab_size).
    Hard constraints are represented by one-hot rows; forbidden values by zeros.
    """

    def __init__(self, transition_matrix: np.ndarray, *, atol: float = 0.0):
        matrix = np.asarray(transition_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition_matrix must be a square 2D array")
        if np.any(matrix < 0):
            raise ValueError("transition_matrix cannot contain negative weights")

        self.transition_matrix = matrix
        self.vocab_size = matrix.shape[0]
        self.atol = atol
        self.outgoing = self._build_outgoing(matrix, atol)
        self.incoming = self._build_incoming(matrix, atol)

    @staticmethod
    def _build_outgoing(matrix: np.ndarray, atol: float):
        result = []
        for prev in range(matrix.shape[0]):
            next_indices = np.flatnonzero(matrix[prev] > atol)
            result.append((next_indices, matrix[prev, next_indices]))
        return result

    @staticmethod
    def _build_incoming(matrix: np.ndarray, atol: float):
        result = []
        for nxt in range(matrix.shape[1]):
            prev_indices = np.flatnonzero(matrix[:, nxt] > atol)
            result.append((prev_indices, matrix[prev_indices, nxt]))
        return result

    @staticmethod
    def _normalize(row: np.ndarray) -> np.ndarray:
        total = row.sum()
        if total <= 0:
            raise NoSolutionErrorInChainSolver("No feasible chain satisfies the constraints.")
        return row / total

    def forward_backward(self, unary_potentials: np.ndarray) -> ForwardBackwardResult:
        unary = np.asarray(unary_potentials, dtype=float)
        if unary.ndim != 2:
            raise ValueError("unary_potentials must be a 2D array")
        if unary.shape[1] != self.vocab_size:
            raise ValueError(
                f"unary_potentials has vocab size {unary.shape[1]}, expected {self.vocab_size}"
            )
        if np.any(unary < 0):
            raise ValueError("unary_potentials cannot contain negative weights")

        length = unary.shape[0]
        if length == 0:
            empty = np.zeros((0, self.vocab_size), dtype=float)
            return ForwardBackwardResult(empty, empty, empty)

        forward = np.zeros_like(unary)
        backward = np.zeros_like(unary)

        forward[0] = self._normalize(unary[0].copy())
        for t in range(1, length):
            row = np.zeros(self.vocab_size, dtype=float)
            previous = forward[t - 1]
            for prev, previous_weight in enumerate(previous):
                if previous_weight <= 0:
                    continue
                next_indices, weights = self.outgoing[prev]
                row[next_indices] += previous_weight * weights
            row *= unary[t]
            forward[t] = self._normalize(row)

        backward[-1] = self._normalize(np.ones(self.vocab_size, dtype=float))
        for t in range(length - 2, -1, -1):
            row = np.zeros(self.vocab_size, dtype=float)
            next_message = unary[t + 1] * backward[t + 1]
            for nxt, next_weight in enumerate(next_message):
                if next_weight <= 0:
                    continue
                prev_indices, weights = self.incoming[nxt]
                row[prev_indices] += weights * next_weight
            backward[t] = self._normalize(row)

        marginals = forward * backward
        row_sums = marginals.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise NoSolutionErrorInChainSolver("No feasible chain satisfies the constraints.")
        marginals = marginals / row_sums
        return ForwardBackwardResult(marginals=marginals, forward=forward, backward=backward)

    def reachable_to_target(self, target_index: int, max_steps: int) -> np.ndarray:
        """
        Compute reachability to a target within an exact number of transitions.

        result[steps, state] is True iff `target_index` can be reached from
        `state` in exactly `steps` Markov transitions with non-zero probability.
        """
        if target_index < 0 or target_index >= self.vocab_size:
            raise IndexError(f"target index {target_index} is outside vocab size {self.vocab_size}")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        reachable = np.zeros((max_steps + 1, self.vocab_size), dtype=bool)
        reachable[0, target_index] = True
        for steps in range(1, max_steps + 1):
            next_reachable = reachable[steps - 1]
            current = reachable[steps]
            for state, (next_indices, _) in enumerate(self.outgoing):
                if np.any(next_reachable[next_indices]):
                    current[state] = True
        return reachable

    def first_hit_reachable_to_target(self, target_index: int, max_steps: int) -> np.ndarray:
        """
        Compute first-hit reachability to a target.

        result[steps, state] is True iff `target_index` can be reached from
        `state` in exactly `steps` transitions without visiting the target
        earlier. This is useful when the target is a stopping state rather than
        an ordinary state that may be traversed or padded through.
        """
        if target_index < 0 or target_index >= self.vocab_size:
            raise IndexError(f"target index {target_index} is outside vocab size {self.vocab_size}")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        reachable = np.zeros((max_steps + 1, self.vocab_size), dtype=bool)
        reachable[0, target_index] = True
        for steps in range(1, max_steps + 1):
            previous = reachable[steps - 1]
            current = reachable[steps]
            for state, (next_indices, _) in enumerate(self.outgoing):
                if state == target_index:
                    continue
                if np.any(previous[next_indices]):
                    current[state] = True
        return reachable

    @staticmethod
    def can_reach_between(reachable: np.ndarray, state_index: int, min_steps: int, max_steps: int) -> bool:
        if min_steps < 0:
            min_steps = 0
        max_steps = min(max_steps, reachable.shape[0] - 1)
        if max_steps < min_steps:
            return False
        return bool(np.any(reachable[min_steps:max_steps + 1, state_index]))


def make_unary_potentials(
    length: int,
    vocab_size: int,
    *,
    forbidden_indices: set[int] | None = None,
    allowed_indices_by_position: dict[int, set[int]] | None = None,
    constraints: dict[int, int] | None = None,
) -> np.ndarray:
    """
    Build unary potentials matching Variable_order_Markov.build_bp_graph().

    Unconstrained positions are uniform over all states except forbidden indices.
    Constrained positions are one-hot, even if the constrained value is normally
    forbidden. This matches PGM.set_value(), which replaces the unary factor.
    """

    if length < 0:
        raise ValueError("length must be non-negative")
    forbidden_indices = forbidden_indices or set()
    allowed_indices_by_position = allowed_indices_by_position or {}
    constraints = constraints or {}

    unary = np.ones((length, vocab_size), dtype=float)
    for index in forbidden_indices:
        unary[:, index] = 0.0

    row_sums = unary.sum(axis=1, keepdims=True)
    if length > 0 and np.any(row_sums <= 0):
        raise ValueError("All vocabulary values are forbidden.")
    np.divide(unary, row_sums, out=unary, where=row_sums > 0)

    for position, allowed_indices in allowed_indices_by_position.items():
        if position < 0 or position >= length:
            raise IndexError(f"constraint position {position} is outside length {length}")
        if not allowed_indices:
            raise ValueError(f"position {position} has an empty allowed set")
        row = np.zeros(vocab_size, dtype=float)
        for value_index in allowed_indices:
            if value_index < 0 or value_index >= vocab_size:
                raise IndexError(f"constraint value {value_index} is outside vocab size {vocab_size}")
            row[value_index] = 1.0
        row_sum = row.sum()
        unary[position] = row / row_sum

    for position, value_index in constraints.items():
        if position < 0 or position >= length:
            raise IndexError(f"constraint position {position} is outside length {length}")
        if value_index < 0 or value_index >= vocab_size:
            raise IndexError(f"constraint value {value_index} is outside vocab size {vocab_size}")
        unary[position] = 0.0
        unary[position, value_index] = 1.0

    return unary
