from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ctor.context_bp.context_graph import ContextGraph


class NoFeasibleSequenceError(Exception):
    """Raised when no path through the context graph satisfies constraints."""


@dataclass(frozen=True)
class ContextBPResult:
    forward: np.ndarray
    backward: np.ndarray
    symbol_marginals: np.ndarray
    path_mass: float


def forward_backward(
    graph: ContextGraph,
    *,
    initial_state: int,
    length: int,
    allowed_symbols_by_position: list[set[int]],
    vocab_size: int,
) -> ContextBPResult:
    if length < 0:
        raise ValueError("length must be non-negative")
    if len(allowed_symbols_by_position) != length:
        raise ValueError("allowed_symbols_by_position must match length")

    n_states = len(graph.contexts)
    forward = np.zeros((length + 1, n_states), dtype=float)
    backward = np.zeros((length + 1, n_states), dtype=float)
    forward[0, initial_state] = 1.0
    path_mass = 1.0

    for position in range(length):
        allowed = allowed_symbols_by_position[position]
        for state, state_weight in enumerate(forward[position]):
            if state_weight <= 0:
                continue
            for edge in graph.outgoing[state]:
                if edge.symbol in allowed:
                    forward[position + 1, edge.dst] += state_weight * edge.weight
        total = forward[position + 1].sum()
        if total <= 0:
            raise NoFeasibleSequenceError("No context path satisfies the constraints.")
        forward[position + 1] /= total
        path_mass *= float(total)

    backward[length] = 1.0
    for position in range(length - 1, -1, -1):
        allowed = allowed_symbols_by_position[position]
        for state, edges in enumerate(graph.outgoing):
            total = 0.0
            for edge in edges:
                if edge.symbol in allowed:
                    total += edge.weight * backward[position + 1, edge.dst]
            backward[position, state] = total
        scale = backward[position].sum()
        if scale <= 0:
            raise NoFeasibleSequenceError("No context path satisfies the constraints.")
        backward[position] /= scale

    if backward[0, initial_state] <= 0:
        raise NoFeasibleSequenceError("No context path satisfies the constraints.")

    symbol_marginals = np.zeros((length, vocab_size), dtype=float)
    for position in range(length):
        allowed = allowed_symbols_by_position[position]
        for state, state_weight in enumerate(forward[position]):
            if state_weight <= 0:
                continue
            for edge in graph.outgoing[state]:
                if edge.symbol in allowed:
                    symbol_marginals[position, edge.symbol] += (
                        state_weight * edge.weight * backward[position + 1, edge.dst]
                    )
        total = symbol_marginals[position].sum()
        if total <= 0:
            raise NoFeasibleSequenceError("No context path satisfies the constraints.")
        symbol_marginals[position] /= total

    return ContextBPResult(
        forward=forward,
        backward=backward,
        symbol_marginals=symbol_marginals,
        path_mass=path_mass,
    )


def backward_messages(
    graph: ContextGraph,
    *,
    length: int,
    allowed_symbols_by_position: list[set[int]],
) -> np.ndarray:
    if length < 0:
        raise ValueError("length must be non-negative")
    if len(allowed_symbols_by_position) != length:
        raise ValueError("allowed_symbols_by_position must match length")

    n_states = len(graph.contexts)
    backward = np.zeros((length + 1, n_states), dtype=float)
    backward[length] = 1.0

    for position in range(length - 1, -1, -1):
        allowed = allowed_symbols_by_position[position]
        for state, edges in enumerate(graph.outgoing):
            total = 0.0
            for edge in edges:
                if edge.symbol in allowed:
                    total += edge.weight * backward[position + 1, edge.dst]
            backward[position, state] = total
        scale = backward[position].sum()
        if scale > 0:
            backward[position] /= scale

    return backward
